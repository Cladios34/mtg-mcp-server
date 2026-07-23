"""Deck simulation workflows -- hypergeometric draw odds and opening hand Monte Carlo.

Pure async functions with no MCP awareness. ``hand_probability`` is a closed-form
calculation with no client dependency. ``simulate_opening_hands`` resolves a decklist
bulk-data-first, classifies each card into a mana-simulation category, then runs a
Monte Carlo goldfish simulation (Commander free mulligan, greedy ramp casting) to
estimate keep rates, land distribution, and mana curve. The default keep rule judges
a hand by a 3-turn goldfish of the hand alone (development, flood, gas); the legacy
lands-only rule is available as ``keep_rule="lands_v1"``. The workflow server
(``server.py``) registers both as MCP tools and handles ToolError conversion.
"""

from __future__ import annotations

import asyncio
import enum
import math
import random
import re
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import structlog

from mtg_mcp_server.utils.color_identity import parse_color_identity
from mtg_mcp_server.utils.decklist import parse_decklist
from mtg_mcp_server.utils.mana import count_pips
from mtg_mcp_server.workflows import WorkflowResult

if TYPE_CHECKING:
    from mtg_mcp_server.services.scryfall import ScryfallClient
    from mtg_mcp_server.services.scryfall_bulk import ScryfallBulkClient
    from mtg_mcp_server.types import Card

log = structlog.get_logger(service="workflow.simulation")

_MANA_SYMBOL_RE = re.compile(r"\{[WUBRGC]\}")
_RAMP_SPELL_RE = re.compile(
    r"search your library for .{0,60}land .{0,80}onto the battlefield",
    re.IGNORECASE | re.DOTALL,
)

# Color-aware simulation (v3) parsing constants.
_COLOR_LETTERS = frozenset("WUBRG")
# Any "search your library" effect: the coarse tutor signal, and the marker that a
# land is a fetch (typed fetchlands say "for a Mountain or Plains card", no "land" word).
_TUTOR_SEARCH_RE = re.compile(r"search your library for", re.IGNORECASE)
# Basic land subtypes -> the mana color they tap for.
_BASIC_TO_COLOR = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G"}
_BASIC_SUBTYPES = frozenset(_BASIC_TO_COLOR)

# Tutor parsing (v3). Color words that qualify a creature tutor ("green creature card").
_TUTOR_COLOR_WORDS = frozenset({"white", "blue", "black", "red", "green"})
# Creature subtypes a tutor may name (case preserved so target matching stays exact).
_TUTOR_SUBTYPES = frozenset(
    {"Elf", "Goblin", "Dragon", "Angel", "Zombie", "Wizard", "Cleric", "Merfolk", "Rebel", "Ally"}
)


# ---------------------------------------------------------------------------
# Hypergeometric probability (hand_probability tool)
# ---------------------------------------------------------------------------


def _hypergeom_pmf(population: int, successes: int, draws: int, observed: int) -> float:
    """Hypergeometric PMF: P(exactly ``observed`` successes in ``draws`` draws).

    ``population`` = N (deck size), ``successes`` = K (matching cards),
    ``draws`` = n (cards seen), ``observed`` = k.
    """
    if observed < 0 or observed > successes:
        return 0.0
    other_draws = draws - observed
    if other_draws < 0 or other_draws > population - successes:
        return 0.0
    return (
        math.comb(successes, observed)
        * math.comb(population - successes, other_draws)
        / math.comb(population, draws)
    )


async def hand_probability(
    deck_size: int = 99,
    copies: int = 1,
    cards_seen: int = 7,
    min_count: int = 1,
    max_count: int | None = None,
) -> WorkflowResult:
    """Compute the exact hypergeometric probability of seeing a card category.

    Answers questions like "what are the odds I've seen at least 1 of my 3
    tutors by turn 4?" via the closed-form hypergeometric distribution -- no
    simulation involved.

    Args:
        deck_size: Total cards in the library (N). Defaults to 99 (Commander).
        copies: Number of matching cards in the deck (K).
        cards_seen: Cards drawn/seen so far (n). Defaults to 7 (opening hand).
        min_count: Minimum matching cards to count as a hit (inclusive).
        max_count: Maximum matching cards to count as a hit (inclusive).
            Defaults to the largest possible count (``min(cards_seen, copies)``).

    Returns:
        WorkflowResult with a PMF table and the requested cumulative probability.
    """
    log.info("hand_probability.start", deck_size=deck_size, copies=copies, cards_seen=cards_seen)

    for label, value in (
        ("deck_size", deck_size),
        ("copies", copies),
        ("cards_seen", cards_seen),
        ("min_count", min_count),
    ):
        if value < 0:
            raise ValueError(f"{label} must be non-negative, got {value}")
    if max_count is not None and max_count < 0:
        raise ValueError(f"max_count must be non-negative, got {max_count}")
    if copies > deck_size:
        raise ValueError(f"copies ({copies}) cannot exceed deck_size ({deck_size})")
    if cards_seen > deck_size:
        raise ValueError(f"cards_seen ({cards_seen}) cannot exceed deck_size ({deck_size})")
    if max_count is not None and max_count < min_count:
        raise ValueError(f"max_count ({max_count}) cannot be less than min_count ({min_count})")

    upper = min(max_count if max_count is not None else cards_seen, cards_seen, copies)
    pmf_range = range(0, min(cards_seen, copies) + 1)
    pmf = {k: _hypergeom_pmf(deck_size, copies, cards_seen, k) for k in pmf_range}
    probability = sum(p for k, p in pmf.items() if min_count <= k <= upper)
    expectation = (cards_seen * copies / deck_size) if deck_size else 0.0

    lines = [
        "# Hypergeometric Draw Probability",
        "",
        f"**Deck Size:** {deck_size}  **Copies:** {copies}  **Cards Seen:** {cards_seen}",
        f"**Expectation:** {expectation:.2f} copies seen on average",
        "",
        "## Probability Mass Function",
        "",
        "| Copies Seen (k) | Probability |",
        "|---|---|",
    ]
    for k in sorted(pmf):
        lines.append(f"| {k} | {pmf[k]:.2%} |")
    lines.append("")
    lines.append(f"**P({min_count} <= copies seen <= {upper}):** {probability:.2%}")

    log.info("hand_probability.complete", probability=probability)
    data: dict[str, object] = {
        "deck_size": deck_size,
        "copies": copies,
        "cards_seen": cards_seen,
        "min_count": min_count,
        "max_count": upper,
        "probability": probability,
        "expectation": expectation,
        "pmf": {str(k): p for k, p in pmf.items()},
    }
    return WorkflowResult(markdown="\n".join(lines), data=data)


# ---------------------------------------------------------------------------
# Card classification (simulate_opening_hands tool)
# ---------------------------------------------------------------------------


class _CardClass(enum.Enum):
    """Mana-simulation category a card is bucketed into."""

    LAND = "land"
    MDFC_LAND = "mdfc_land"
    ROCK = "rock"
    DORK = "dork"
    RAMP_SPELL = "ramp_spell"
    OTHER = "other"


_RAMP_CLASSES = frozenset({_CardClass.ROCK, _CardClass.DORK, _CardClass.RAMP_SPELL})


class _Slot(NamedTuple):
    """A single deck slot reduced to only what the simulation loop needs.

    The trailing ``colors``/``pips``/``is_tutor`` fields default to empty so
    positional constructions (``_Slot(cls, cmc, prod)``) stay valid; they are
    only populated when the color-screen or tutor-analysis features are active.
    """

    cls: _CardClass
    cmc: int
    production: int
    colors: frozenset[str] = frozenset()
    pips: frozenset[str] = frozenset()
    is_tutor: bool = False


def _classify_card(
    card: Card,
    *,
    extra_mana_sources: frozenset[str],
    exclude_cards: frozenset[str],
) -> _CardClass:
    """Classify a resolved card into a mana-simulation category.

    Rules are checked in order; the first match wins.
    """
    name_lower = card.name.lower()
    if name_lower in exclude_cards:
        return _CardClass.OTHER
    if name_lower in extra_mana_sources:
        return _CardClass.ROCK

    front_type = card.type_line.split(" // ")[0]
    if "Land" in front_type:
        return _CardClass.LAND

    if " // " in card.type_line:
        _, back_type = card.type_line.split(" // ", 1)
        if "Land" in back_type:
            return _CardClass.MDFC_LAND

    oracle = card.oracle_text or ""
    if (
        "Artifact" in card.type_line
        and "Creature" not in card.type_line
        and card.cmc <= 3
        and "{T}: Add" in oracle
    ):
        return _CardClass.ROCK
    if "Creature" in card.type_line and card.cmc <= 3 and "{T}: Add" in oracle:
        return _CardClass.DORK
    if _RAMP_SPELL_RE.search(oracle):
        return _CardClass.RAMP_SPELL
    return _CardClass.OTHER


def _mana_production(card: Card | None, cls: _CardClass) -> int:
    """Compute how much mana a classified card produces when active.

    Lands, MDFC lands, and ramp-spell lands produce 1. Rocks and dorks produce
    the number of colored/colorless mana symbols in the first oracle line that
    contains "Add" (minimum 1). A rock/dork with no resolved card data (an
    ``extra_mana_sources`` override on an unresolved name) defaults to 1.
    """
    if cls in (_CardClass.LAND, _CardClass.MDFC_LAND, _CardClass.RAMP_SPELL):
        return 1
    if cls in (_CardClass.ROCK, _CardClass.DORK):
        if card is None:
            return 1
        oracle = card.oracle_text or ""
        for line in oracle.splitlines():
            if "Add" in line:
                return max(len(_MANA_SYMBOL_RE.findall(line)), 1)
        return 1
    return 0


def _fetch_colors(card: Card, deck_land_colors: frozenset[str]) -> frozenset[str]:
    """Colors a land-fetch effect resolves to, deck-aware.

    If the oracle names specific basic land subtypes (e.g. a fetchland that
    grabs a "Mountain or Plains"), return those subtypes' colors. Otherwise
    (a generic "search for a basic land" ramp spell) fall back to the union of
    every non-fetch land color the deck actually runs.
    """
    oracle = card.oracle_text or ""
    named = frozenset(color for sub, color in _BASIC_TO_COLOR.items() if sub in oracle)
    return named or deck_land_colors


def _source_colors(
    card: Card, cls: _CardClass, *, deck_land_colors: frozenset[str]
) -> frozenset[str]:
    """Colors a classified card can add to the mana pool (WUBRG only).

    Non-fetch lands, rocks, and dorks report their own ``produced_mana``;
    fetchlands and land-fetch ramp spells report deck-aware fetch colors;
    ``OTHER`` cards produce no mana, hence no color.
    """
    if cls is _CardClass.OTHER:
        return frozenset()
    if cls is _CardClass.RAMP_SPELL:
        return _fetch_colors(card, deck_land_colors)
    if cls in (_CardClass.LAND, _CardClass.MDFC_LAND) and _TUTOR_SEARCH_RE.search(
        card.oracle_text or ""
    ):
        return _fetch_colors(card, deck_land_colors)
    return frozenset(card.produced_mana) & _COLOR_LETTERS


def _deck_land_colors(
    cards: dict[str, Card], *, extra_mana_sources: frozenset[str], exclude_cards: frozenset[str]
) -> frozenset[str]:
    """Union of colors produced by every non-fetch land resolved in the deck.

    Precomputed once (never inside the Monte Carlo loop) so generic land-fetch
    effects can pin onto whatever lands the deck actually runs.
    """
    colors: set[str] = set()
    for card in cards.values():
        cls = _classify_card(
            card, extra_mana_sources=extra_mana_sources, exclude_cards=exclude_cards
        )
        if cls in (_CardClass.LAND, _CardClass.MDFC_LAND) and not _TUTOR_SEARCH_RE.search(
            card.oracle_text or ""
        ):
            colors |= frozenset(card.produced_mana) & _COLOR_LETTERS
    return frozenset(colors)


class _TutorInfo(NamedTuple):
    """A detected tutor and what it fetches, for the tutor-analysis report."""

    name: str
    target_constraint: str
    destination: str
    speed: str
    target_names: list[str]
    target_count: int


def _tutor_constraint(oracle: str, lower: str) -> str:
    """Best-effort category of what a tutor searches for (first match wins).

    Fixed order: color creature, creature, artifact/enchantment, instant/sorcery,
    named creature subtype, basic land, land, then unconstrained ("any").
    """
    has_creature = "creature card" in lower or "creature spell" in lower
    if has_creature and any(w in lower for w in _TUTOR_COLOR_WORDS):
        return "color_creature"
    if has_creature:
        return "creature"
    if "artifact" in lower or "enchantment" in lower:
        return "artifact_enchantment"
    if "instant" in lower or "sorcery" in lower:
        return "instant_sorcery"
    for sub in _TUTOR_SUBTYPES:
        if sub in oracle:
            return f"subtype:{sub}"
    if "basic land" in lower:
        return "basic_land"
    if "land" in lower:
        return "land"
    return "any"


def _tutor_destination(lower: str) -> str:
    """Where a tutor puts the fetched card (checked in a fixed order)."""
    if "onto the battlefield" in lower:
        return "battlefield"
    if "into your graveyard" in lower:
        return "graveyard"
    # "on top" (not "on top of"): current Scryfall templating shortens to
    # "put that card on top." (seen on Vampiric Tutor / Imperial Seal, 2026-07-23).
    if "on top" in lower:
        return "top"
    if "into your hand" in lower:
        return "hand"
    return "hand"


def _classify_tutor(card: Card, cls: _CardClass) -> tuple[str, str, str] | None:
    """Classify a tutor as ``(constraint, destination, speed)``, or None.

    Lands, MDFC lands, and land-fetch ramp spells are never tutors (excluded by
    class). A tutor must search the library; otherwise returns None.
    """
    if cls in (_CardClass.LAND, _CardClass.MDFC_LAND, _CardClass.RAMP_SPELL):
        return None
    oracle = card.oracle_text or ""
    if not _TUTOR_SEARCH_RE.search(oracle):
        return None
    lower = oracle.lower()
    constraint = _tutor_constraint(oracle, lower)
    destination = _tutor_destination(lower)
    speed = "instant" if "Instant" in card.type_line or "flash" in lower else "sorcery"
    return constraint, destination, speed


def _tutor_matches(constraint: str, card: Card) -> bool:
    """Whether ``card`` is a legal target for a tutor of the given constraint."""
    t = card.type_line
    if constraint in ("creature", "color_creature"):
        return "Creature" in t
    if constraint == "artifact_enchantment":
        return "Artifact" in t or "Enchantment" in t
    if constraint == "instant_sorcery":
        return "Instant" in t or "Sorcery" in t
    if constraint.startswith("subtype:"):
        return constraint.split(":", 1)[1] in t
    if constraint == "basic_land":
        return "Basic" in t and "Land" in t
    if constraint == "land":
        return "Land" in t
    return True  # "any" or unknown: every card qualifies


def _tutor_targets(constraint: str, deck_cards: list[Card]) -> tuple[list[str], int]:
    """Names (sample of up to 12) and total count of a tutor's legal targets."""
    matches = [c.name for c in deck_cards if _tutor_matches(constraint, c)]
    return matches[:12], len(matches)


async def _resolve_deck(
    decklist: list[str],
    *,
    bulk: ScryfallBulkClient | None,
    scryfall: ScryfallClient,
) -> tuple[list[tuple[int, str]], dict[str, Card], list[str]]:
    """Parse and resolve a decklist, bulk-data-first with Scryfall fallback.

    Returns:
        A tuple of ``(slots, cards_by_name, unresolved)`` where ``slots`` is
        the parsed ``(quantity, name)`` pairs in decklist order, ``cards_by_name``
        maps lowercase card name to the resolved ``Card``, and ``unresolved``
        lists the original names that could not be resolved.
    """
    from mtg_mcp_server.workflows.card_resolver import resolve_card

    slots = parse_decklist(decklist)
    unique_names = list(dict.fromkeys(name for _, name in slots))

    # Cap concurrent Scryfall lookups to avoid overwhelming the connection pool.
    sem = asyncio.Semaphore(10)

    async def _bounded_resolve(name: str) -> Card:
        async with sem:
            return await resolve_card(name, bulk=bulk, scryfall=scryfall)

    tasks = [_bounded_resolve(name) for name in unique_names]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    cards_by_name: dict[str, Card] = {}
    unresolved: list[str] = []
    for name, result in zip(unique_names, results, strict=True):
        if isinstance(result, BaseException):
            log.warning(
                "simulate_opening_hands.resolve_failed",
                card=name,
                error=str(result),
                error_type=type(result).__name__,
            )
            unresolved.append(name)
        else:
            cards_by_name[name.lower()] = result

    return slots, cards_by_name, unresolved


# ---------------------------------------------------------------------------
# Monte Carlo goldfish engine -- pure, synchronous, hot-loop friendly
# ---------------------------------------------------------------------------


def _effective_lands(hand: list[_Slot], count_mdfc_lands: bool) -> float:
    """Effective land count: LAND cards plus half of MDFC lands (if counted)."""
    lands = float(sum(1 for c in hand if c.cls == _CardClass.LAND))
    if count_mdfc_lands:
        lands += 0.5 * sum(1 for c in hand if c.cls == _CardClass.MDFC_LAND)
    return lands


def _keep_reason_lands_v1(
    hand: list[_Slot], *, min_lands: int, max_lands: int, count_mdfc_lands: bool
) -> str | None:
    """Legacy keep rule: effective lands within [min_lands, max_lands] inclusive.

    Weighs lands only (not rocks/dorks/curve). Preserved as ``keep_rule="lands_v1"``
    for exact reproduction of pre-playability-rule results. Returns ``None`` if
    the hand is keepable, otherwise the reason it was not.
    """
    effective = _effective_lands(hand, count_mdfc_lands)
    if effective < min_lands:
        return "screw"
    if effective > max_lands:
        return "flood"
    return None


def _hand_color_sources(hand: list[_Slot]) -> frozenset[str]:
    """Union of the colors every mana source in the hand can produce."""
    sources: set[str] = set()
    for c in hand:
        sources |= c.colors
    return frozenset(sources)


def _keep_reason_colors(hand: list[_Slot], *, commander_colors: frozenset[str]) -> str | None:
    """Color-screen check: fail if the hand cannot source every commander color.

    Inert (always ``None``) when ``commander_colors`` is empty, so the color
    screen only fires when the caller opts in via ``commander_colors``.
    """
    if not commander_colors:
        return None
    if commander_colors <= _hand_color_sources(hand):
        return None
    return "colors"


def _keep_playability(
    hand: list[_Slot],
    *,
    min_lands: int,
    max_lands: int,
    count_mdfc_lands: bool,
    gas_cmc_threshold: int,
    commander_colors: frozenset[str] = frozenset(),
    tutor_aware: bool = False,
) -> str | None:
    """Playability keep rule: hand development speed, flood, castable gas, colors.

    Checked in this fixed order, the first failing check wins:

    1. "screw" -- the hand's 3-turn goldfish (:func:`_hand_development`)
       produces less mana on turn 3 than ``min_lands``, meaning the hand is
       too slow to develop even accounting for rocks/dorks/ramp spells.
    2. "flood" -- more effective lands than ``max_lands``.
    3. "no_gas" -- no non-land, non-ramp card at or below ``gas_cmc_threshold``
       mana value to actually do something with the mana. Under ``tutor_aware``,
       a tutor at or below the threshold also counts as gas (it can find some).
    4. "colors" -- the hand cannot source every ``commander_colors`` color
       (inert when ``commander_colors`` is empty).

    Returns ``None`` if the hand is keepable.
    """
    if _hand_development(hand, count_mdfc_lands=count_mdfc_lands) < min_lands:
        return "screw"
    if _effective_lands(hand, count_mdfc_lands) > max_lands:
        return "flood"
    has_gas = any(
        (
            c.cls in (_CardClass.OTHER, _CardClass.RAMP_SPELL)
            and not c.is_tutor
            and c.cmc <= gas_cmc_threshold
        )
        or (tutor_aware and c.is_tutor and c.cmc <= gas_cmc_threshold)
        for c in hand
    )
    if not has_gas:
        return "no_gas"
    return _keep_reason_colors(hand, commander_colors=commander_colors)


def _bottom_cards(
    hand: list[_Slot], n: int, count_mdfc_lands: bool
) -> tuple[list[_Slot], list[_Slot]]:
    """Select ``n`` cards to bottom for a London mulligan.

    Bottoms a land while more than 3 effective lands remain in hand, otherwise
    bottoms the highest-CMC non-land card (ties broken by hand order, stable).
    """
    remaining = list(hand)
    bottomed: list[_Slot] = []
    for _ in range(n):
        if not remaining:
            break
        effective = _effective_lands(remaining, count_mdfc_lands)
        land_indices = [
            i for i, c in enumerate(remaining) if c.cls in (_CardClass.LAND, _CardClass.MDFC_LAND)
        ]
        if effective > 3 and land_indices:
            idx = land_indices[0]
        else:
            non_land_indices = [
                i
                for i, c in enumerate(remaining)
                if c.cls not in (_CardClass.LAND, _CardClass.MDFC_LAND)
            ]
            if non_land_indices:
                idx = max(non_land_indices, key=lambda i: remaining[i].cmc)
            elif land_indices:
                idx = land_indices[0]
            else:
                idx = 0
        bottomed.append(remaining.pop(idx))
    return remaining, bottomed


def _is_sole_source(remaining: list[_Slot], idx: int, commander_colors: frozenset[str]) -> bool:
    """True if slot ``idx`` is the only card sourcing some commander color.

    Always False when ``commander_colors`` is empty, so color-aware bottoming is
    a no-op unless the caller opts in.
    """
    unique = remaining[idx].colors & commander_colors
    if not unique:
        return False
    others: set[str] = set()
    for i, c in enumerate(remaining):
        if i != idx:
            others |= c.colors
    return not (unique <= others)


def _bottom_cards_playability(
    hand: list[_Slot],
    n: int,
    *,
    count_mdfc_lands: bool,
    max_lands: int,
    gas_cmc_threshold: int,
    commander_colors: frozenset[str] = frozenset(),
) -> tuple[list[_Slot], list[_Slot]]:
    """Select ``n`` cards to bottom for a London mulligan (playability rule).

    Each of the ``n`` iterations bottoms exactly one slot, in this fixed
    priority order:

    1. Trim excess lands: while more than ``min(3, max_lands)`` effective
       lands remain, bottom the first LAND (or the first MDFC_LAND if none).
    2. Otherwise, protect the cheapest gas card (OTHER/RAMP_SPELL at or below
       ``gas_cmc_threshold``, first on ties) and bottom the most expensive
       OTHER/RAMP_SPELL card that is not the protected one.
    3. Otherwise, bottom the most expensive ROCK/DORK.
    4. Otherwise, bottom the first remaining land, then the protected gas
       card, then whatever is left at index 0.

    When ``commander_colors`` is set, a land that is the sole source of a
    commander color is protected from the land-cut branches; if every candidate
    land is a sole source, the branch falls back to its default pick (never stalls).
    """
    remaining = list(hand)
    bottomed: list[_Slot] = []
    land_cap = min(3.0, float(max_lands))
    for _ in range(n):
        if not remaining:
            break

        land_indices = [
            i for i, c in enumerate(remaining) if c.cls in (_CardClass.LAND, _CardClass.MDFC_LAND)
        ]
        pure_land_indices = [i for i in land_indices if remaining[i].cls == _CardClass.LAND]
        effective = _effective_lands(remaining, count_mdfc_lands)

        if effective > land_cap and land_indices:
            keep_pure = [
                i for i in pure_land_indices if not _is_sole_source(remaining, i, commander_colors)
            ]
            keep_any = [
                i for i in land_indices if not _is_sole_source(remaining, i, commander_colors)
            ]
            if keep_pure:
                idx = keep_pure[0]
            elif keep_any:
                idx = keep_any[0]
            else:
                idx = pure_land_indices[0] if pure_land_indices else land_indices[0]
            bottomed.append(remaining.pop(idx))
            continue

        gas_indices = [
            i
            for i, c in enumerate(remaining)
            if c.cls in (_CardClass.OTHER, _CardClass.RAMP_SPELL) and c.cmc <= gas_cmc_threshold
        ]
        protected_idx = min(gas_indices, key=lambda i: remaining[i].cmc) if gas_indices else None

        gas_like_indices = [
            i
            for i, c in enumerate(remaining)
            if c.cls in (_CardClass.OTHER, _CardClass.RAMP_SPELL) and i != protected_idx
        ]
        if gas_like_indices:
            idx = max(gas_like_indices, key=lambda i: remaining[i].cmc)
            bottomed.append(remaining.pop(idx))
            continue

        ramp_indices = [
            i for i, c in enumerate(remaining) if c.cls in (_CardClass.ROCK, _CardClass.DORK)
        ]
        if ramp_indices:
            idx = max(ramp_indices, key=lambda i: remaining[i].cmc)
            bottomed.append(remaining.pop(idx))
            continue

        keep_lands = [
            i for i in land_indices if not _is_sole_source(remaining, i, commander_colors)
        ]
        if keep_lands:
            idx = keep_lands[0]
        elif land_indices:
            idx = land_indices[0]
        elif protected_idx is not None:
            idx = protected_idx
        else:
            idx = 0
        bottomed.append(remaining.pop(idx))

    return remaining, bottomed


def _battlefield_pool(battlefield: list[tuple[_CardClass, int, int]], turn: int) -> int:
    """Sum of active mana production on the battlefield during a given turn.

    Lands, MDFC lands, and rocks produce starting the turn they are played;
    dorks and ramp spells (summoning sickness / enters tapped) start
    producing the following turn.
    """
    pool = 0
    for cls, production, played_turn in battlefield:
        if cls in (_CardClass.LAND, _CardClass.MDFC_LAND, _CardClass.ROCK):
            if played_turn <= turn:
                pool += production
        elif played_turn < turn:
            pool += production
    return pool


def _hand_development(hand: list[_Slot], *, count_mdfc_lands: bool) -> float:
    """Mini-goldfish of the opening hand alone (no draws) across turns 1-3.

    Mirrors the greedy land-drop and ramp-casting logic of ``_goldfish`` but
    with no library draws, since the playability keep rule judges the hand
    before it sees a mulligan decision. Returns the raw (uncast) mana pool
    available on turn 3, before casting anything that turn -- a proxy for how
    fast the hand develops, used by the playability keep rule.
    """
    remaining = list(hand)
    battlefield: list[tuple[_CardClass, int, int]] = []

    for turn in range(1, 4):
        land_idx = next((i for i, c in enumerate(remaining) if c.cls == _CardClass.LAND), None)
        if land_idx is None and count_mdfc_lands:
            land_idx = next(
                (i for i, c in enumerate(remaining) if c.cls == _CardClass.MDFC_LAND), None
            )
        if land_idx is not None:
            played = remaining.pop(land_idx)
            battlefield.append((played.cls, played.production, turn))

        pool = _battlefield_pool(battlefield, turn)

        if turn == 3:
            return float(pool)

        castable = sorted((c for c in remaining if c.cls in _RAMP_CLASSES), key=lambda c: c.cmc)
        for card in castable:
            if card.cmc > pool:
                break
            pool -= card.cmc
            remaining.remove(card)
            battlefield.append((card.cls, card.production, turn))
            if card.cls == _CardClass.ROCK:
                pool += card.production

    raise AssertionError("unreachable: range(1, 4) always returns at turn == 3")


def _goldfish(
    kept_hand: list[_Slot],
    library: list[_Slot],
    *,
    count_mdfc_lands: bool,
) -> dict[str, Any]:
    """Greedy goldfish over turns 1-5: 1 land drop then castable ramp pieces.

    Timing: lands and mana rocks produce immediately on the turn played. Dorks
    (summoning sickness) and ramp-spell lands (enter tapped) become active
    starting the following turn. The turn's spendable mana is the pool left
    over after casting this turn's payable rocks/dorks/ramp spells.
    """
    hand = list(kept_hand)
    draw_pool = list(library)
    battlefield: list[tuple[_CardClass, int, int]] = []  # (cls, production, played_turn)
    spendable_by_turn: list[float] = []
    ramp_by_turn2 = False

    for turn in range(1, 6):
        if draw_pool:
            hand.append(draw_pool.pop(0))

        land_idx = next((i for i, c in enumerate(hand) if c.cls == _CardClass.LAND), None)
        if land_idx is None and count_mdfc_lands:
            land_idx = next((i for i, c in enumerate(hand) if c.cls == _CardClass.MDFC_LAND), None)
        if land_idx is not None:
            played = hand.pop(land_idx)
            battlefield.append((played.cls, played.production, turn))

        pool = _battlefield_pool(battlefield, turn)

        castable = sorted((c for c in hand if c.cls in _RAMP_CLASSES), key=lambda c: c.cmc)
        for card in castable:
            if card.cmc > pool:
                break
            pool -= card.cmc
            hand.remove(card)
            battlefield.append((card.cls, card.production, turn))
            if card.cls == _CardClass.ROCK:
                pool += card.production
            if turn <= 2:
                ramp_by_turn2 = True

        spendable_by_turn.append(float(pool))

    return {"spendable_by_turn": spendable_by_turn, "ramp_by_turn2": ramp_by_turn2}


def _simulate_once(
    deck: list[_Slot],
    rng: random.Random,
    *,
    min_lands: int,
    max_lands: int,
    count_mdfc_lands: bool,
    keep_rule: Literal["playability", "lands_v1"],
    free_mulligan: bool,
    gas_cmc_threshold: int,
    commander_colors: frozenset[str] = frozenset(),
    tutor_aware: bool = False,
) -> dict[str, Any]:
    """Simulate one London-mulligan opening hand plus a 5-turn goldfish.

    ``free_mulligan`` applies the Commander free mulligan: the first
    mulligan redraws a full 7 without bottoming, only the second and later
    mulligans bottom a card. ``keep_rule`` selects between the playability
    rule (hand development, flood, gas -- see :func:`_keep_playability`) and
    the legacy lands-only rule (:func:`_keep_reason_lands_v1`). ``commander_colors``
    and ``tutor_aware`` drive the color screen and tutor-as-gas extensions; both
    are inert by default so the legacy behavior is unchanged.
    """
    library = list(deck)
    rng.shuffle(library)

    mulligans = 0
    kept_hand: list[_Slot] = []
    rest: list[_Slot] = []
    bottomed: list[_Slot] = []
    mull_reasons: list[str] = []
    while True:
        hand = library[:7]
        rest = library[7:]
        bottoms = max(0, mulligans - 1) if free_mulligan else mulligans
        if keep_rule == "playability":
            kept_candidate, bottomed = _bottom_cards_playability(
                hand,
                bottoms,
                count_mdfc_lands=count_mdfc_lands,
                max_lands=max_lands,
                gas_cmc_threshold=gas_cmc_threshold,
                commander_colors=commander_colors,
            )
            keep_reason = _keep_playability(
                kept_candidate,
                min_lands=min_lands,
                max_lands=max_lands,
                count_mdfc_lands=count_mdfc_lands,
                gas_cmc_threshold=gas_cmc_threshold,
                commander_colors=commander_colors,
                tutor_aware=tutor_aware,
            )
        else:
            kept_candidate, bottomed = _bottom_cards(hand, bottoms, count_mdfc_lands)
            keep_reason = _keep_reason_lands_v1(
                kept_candidate,
                min_lands=min_lands,
                max_lands=max_lands,
                count_mdfc_lands=count_mdfc_lands,
            )
        forced = bottoms >= 3
        if forced or keep_reason is None:
            kept_hand = kept_candidate
            break
        mull_reasons.append(keep_reason)
        # Mulligan again: shuffle the drawn hand back into the library (London mulligan).
        library = rest + hand
        rng.shuffle(library)
        mulligans += 1

    final_library = rest + bottomed
    goldfish = _goldfish(kept_hand, final_library, count_mdfc_lands=count_mdfc_lands)

    return {
        "hand_size": len(kept_hand),
        "effective_lands": _effective_lands(kept_hand, count_mdfc_lands),
        "spendable_by_turn": goldfish["spendable_by_turn"],
        "ramp_by_turn2": goldfish["ramp_by_turn2"],
        "mulligans": mulligans,
        "mull_reasons": mull_reasons,
        "has_tutor": any(c.is_tutor for c in kept_hand),
        "kept_hand": kept_hand,
    }


def _run_simulation(
    deck: list[_Slot],
    *,
    iterations: int,
    seed: int | None,
    min_lands: int,
    max_lands: int,
    count_mdfc_lands: bool,
    keep_rule: Literal["playability", "lands_v1"] = "playability",
    free_mulligan: bool = True,
    gas_cmc_threshold: int = 4,
    commander_colors: frozenset[str] = frozenset(),
    tutor_aware: bool = False,
) -> dict[str, Any]:
    """Run the Monte Carlo goldfish simulation ``iterations`` times and aggregate.

    ``commander_colors`` and ``tutor_aware`` are inert by default: the "colors"
    mull-reason count stays 0 and ``tutor_in_hand_pct`` stays 0 unless a caller
    opts into the color-screen / tutor-analysis extensions.

    ``hand_composition`` additionally profiles the KEPT hands (lands, rocks/
    dorks, tutors, gas, top-end, and per-color sourcing), independent of the
    keep/bottoming rules above -- it never changes which hands are kept.
    """
    rng = random.Random(seed)
    keep_counts = {7: 0, 6: 0, 5: 0, 4: 0}
    kept_land_totals: dict[float, int] = {}
    spendable_totals = [0.0] * 5
    on_curve_counts = [0] * 5
    ramp_by_turn2 = 0
    tutor_in_hand = 0
    mulligan_counts = {"0": 0, "1": 0, "2": 0, "3plus": 0}
    mull_reason_counts = {"screw": 0, "flood": 0, "no_gas": 0, "colors": 0}

    # Hand-composition accumulators (T13): computed once per kept hand, never
    # inside the keep/bottoming decision itself.
    deck_source_colors: frozenset[str] = frozenset()
    for slot in deck:
        deck_source_colors |= slot.colors
    hand_lands_total = 0.0
    hand_ramp_total = 0
    hand_tutor_total = 0
    hand_gas_total = 0
    hand_big_total = 0
    no_gas_hands = 0
    big2plus_hands = 0
    color_hit_counts = {color: 0 for color in deck_source_colors}

    for _ in range(iterations):
        result = _simulate_once(
            deck,
            rng,
            min_lands=min_lands,
            max_lands=max_lands,
            count_mdfc_lands=count_mdfc_lands,
            keep_rule=keep_rule,
            free_mulligan=free_mulligan,
            gas_cmc_threshold=gas_cmc_threshold,
            commander_colors=commander_colors,
            tutor_aware=tutor_aware,
        )
        if result["has_tutor"]:
            tutor_in_hand += 1
        keep_counts[result["hand_size"]] += 1
        lands = result["effective_lands"]
        kept_land_totals[lands] = kept_land_totals.get(lands, 0) + 1
        spendable_by_turn = result["spendable_by_turn"]
        for i in range(5):
            spendable_totals[i] += spendable_by_turn[i]
            if spendable_by_turn[i] >= (i + 1):
                on_curve_counts[i] += 1
        if result["ramp_by_turn2"]:
            ramp_by_turn2 += 1
        mulligans = result["mulligans"]
        mulligan_key = str(mulligans) if mulligans < 3 else "3plus"
        mulligan_counts[mulligan_key] += 1
        for reason in result["mull_reasons"]:
            mull_reason_counts[reason] += 1

        kept_hand = result["kept_hand"]
        hand_lands_total += _effective_lands(kept_hand, count_mdfc_lands)
        hand_ramp_total += sum(1 for c in kept_hand if c.cls in (_CardClass.ROCK, _CardClass.DORK))
        if tutor_aware:
            hand_tutor_total += sum(1 for c in kept_hand if c.is_tutor)
        gas_count = sum(
            1 for c in kept_hand if c.cls is _CardClass.OTHER and c.cmc <= gas_cmc_threshold
        )
        hand_gas_total += gas_count
        if gas_count == 0:
            no_gas_hands += 1
        big_count = sum(1 for c in kept_hand if c.cmc >= 6)
        hand_big_total += big_count
        if big_count >= 2:
            big2plus_hands += 1
        hand_colors: frozenset[str] = frozenset()
        for c in kept_hand:
            hand_colors |= c.colors
        for color in deck_source_colors:
            if color in hand_colors:
                color_hit_counts[color] += 1

    hand_composition = {
        "avg_lands": hand_lands_total / iterations,
        "avg_rocks_dorks": hand_ramp_total / iterations,
        "avg_tutors": (hand_tutor_total / iterations) if tutor_aware else None,
        "avg_gas": hand_gas_total / iterations,
        "avg_cmc6plus": hand_big_total / iterations,
        "pct_no_gas": no_gas_hands / iterations,
        "pct_2plus_cmc6": big2plus_hands / iterations,
        "color_source_pct": {
            color: color_hit_counts[color] / iterations for color in sorted(deck_source_colors)
        },
    }

    return {
        "keep_pct_by_hand_size": {size: count / iterations for size, count in keep_counts.items()},
        "kept_land_distribution": {
            lands: count / iterations for lands, count in kept_land_totals.items()
        },
        "avg_spendable_mana_by_turn": {i + 1: spendable_totals[i] / iterations for i in range(5)},
        "on_curve_pct_by_turn": {i + 1: on_curve_counts[i] / iterations for i in range(5)},
        "ramp_by_turn2_pct": ramp_by_turn2 / iterations,
        "tutor_in_hand_pct": tutor_in_hand / iterations,
        "keep_pct_by_mulligans": {k: v / iterations for k, v in mulligan_counts.items()},
        "mull_reasons": mull_reason_counts,
        "hand_composition": hand_composition,
    }


# ---------------------------------------------------------------------------
# simulate_opening_hands tool
# ---------------------------------------------------------------------------


async def simulate_opening_hands(
    decklist: list[str],
    *,
    iterations: int = 10000,
    seed: int | None = None,
    min_lands: int = 2,
    max_lands: int = 5,
    count_mdfc_lands: bool = True,
    keep_rule: Literal["playability", "lands_v1"] = "playability",
    free_mulligan: bool = True,
    gas_cmc_threshold: int = 4,
    extra_mana_sources: list[str] | None = None,
    exclude_cards: list[str] | None = None,
    commander_colors: str | None = None,
    tutor_aware: bool = False,
    bulk: ScryfallBulkClient | None,
    scryfall: ScryfallClient,
) -> WorkflowResult:
    """Monte Carlo simulation of opening hands, mulligans, and early mana curve.

    Resolves the decklist bulk-data-first, classifies each card as a land, MDFC
    land, mana rock, mana dork, land-fetch ramp spell, or other, then simulates
    London-mulligan opening hands and a greedy 5-turn goldfish to estimate keep
    rates, kept-hand land distribution, and spendable mana per turn. The default
    keep rule ("playability") judges a hand by a 3-turn goldfish of the hand
    alone -- development speed, flood, and castable gas -- rather than lands
    alone; the legacy lands-only rule is available as ``keep_rule="lands_v1"``.

    Optional ``commander_colors`` enables a color screen (a hand that cannot
    source every commander color is mulliganed) and ``tutor_aware`` treats cheap
    tutors as gas and reports tutor coverage. The color screen applies to the
    playability keep rule only; ``keep_rule="lands_v1"`` ignores it entirely.

    Args:
        decklist: Card entries, excluding the commander (a 99-card Commander
            library). Lands and spells both belong in this list.
        iterations: Number of simulated games (100-100000).
        seed: RNG seed for reproducible results. Omit for a random seed.
        min_lands: Minimum turn-3 mana pool (playability rule) or minimum
            effective lands (lands_v1 rule) for a hand to avoid a "screw"
            mulligan.
        max_lands: Maximum effective lands to keep an opening hand.
        count_mdfc_lands: Count modal-double-faced land backs as half a land.
        keep_rule: "playability" (3-turn goldfish of the hand: development,
            flood, gas) or "lands_v1" (legacy effective-land range only).
        free_mulligan: Commander free mulligan -- the first mulligan redraws
            a full 7 without bottoming a card; only the second and later
            mulligans bottom one.
        gas_cmc_threshold: A hand needs at least one non-land, non-ramp card
            at or below this mana value to be kept under the playability rule.
        extra_mana_sources: Card names to force-classify as mana rocks (useful
            for cards this tool misclassifies or cannot resolve).
        exclude_cards: Card names to force-classify as non-mana (e.g. false
            positives from the ramp-spell heuristic).
        commander_colors: Commander color identity (e.g. "mardu", "WBR",
            "boros"). Enables the playability-rule color screen. Raises
            ValueError if unparseable.
        tutor_aware: Detect tutors, count them as gas, and report per-tutor
            targets and the odds of an opening hand holding a tutor.
        bulk: Optional ScryfallBulkClient for rate-limit-free resolution.
        scryfall: Initialized ScryfallClient (fallback resolver).

    Returns:
        WorkflowResult with markdown and structured data.
    """
    log.info("simulate_opening_hands.start", cards=len(decklist), iterations=iterations)

    if not (100 <= iterations <= 100000):
        raise ValueError(f"iterations must be between 100 and 100000, got {iterations}")
    if min_lands > max_lands:
        raise ValueError(f"min_lands ({min_lands}) cannot exceed max_lands ({max_lands})")
    if keep_rule not in ("playability", "lands_v1"):
        raise ValueError(f"keep_rule must be 'playability' or 'lands_v1', got {keep_rule!r}")
    if gas_cmc_threshold < 0:
        raise ValueError(f"gas_cmc_threshold must be non-negative, got {gas_cmc_threshold}")

    parsed_colors = parse_color_identity(commander_colors) if commander_colors else frozenset()

    slots, cards_by_name, unresolved = await _resolve_deck(decklist, bulk=bulk, scryfall=scryfall)

    deck_size = sum(qty for qty, _ in slots)
    if deck_size < 7:
        raise ValueError(f"Deck must contain at least 7 cards, got {deck_size}")

    extra_set = frozenset(n.lower() for n in (extra_mana_sources or []))
    exclude_set = frozenset(n.lower() for n in (exclude_cards or []))

    deck_land_colors = _deck_land_colors(
        cards_by_name, extra_mana_sources=extra_set, exclude_cards=exclude_set
    )
    resolved_cards = list(cards_by_name.values())

    card_classes: dict[str, list[str]] = {c.name: [] for c in _CardClass}
    slot_by_key: dict[str, _Slot] = {}
    tutors: list[_TutorInfo] = []
    for name in dict.fromkeys(name for _, name in slots):
        key = name.lower()
        card = cards_by_name.get(key)
        is_tutor = False
        if card is not None:
            cls = _classify_card(card, extra_mana_sources=extra_set, exclude_cards=exclude_set)
            cmc_int = round(card.cmc)
            display_name = card.name
            colors = _source_colors(card, cls, deck_land_colors=deck_land_colors)
            pips = frozenset(count_pips(card.mana_cost)) & _COLOR_LETTERS
            if tutor_aware:
                classified = _classify_tutor(card, cls)
                if classified is not None:
                    is_tutor = True
                    constraint, destination, speed = classified
                    target_names, target_count = _tutor_targets(constraint, resolved_cards)
                    tutors.append(
                        _TutorInfo(
                            display_name, constraint, destination, speed, target_names, target_count
                        )
                    )
        else:
            cls = (
                _CardClass.ROCK if key in extra_set and key not in exclude_set else _CardClass.OTHER
            )
            cmc_int = 0
            display_name = name
            colors = frozenset()
            pips = frozenset()
        production = _mana_production(card, cls)
        slot_by_key[key] = _Slot(cls, cmc_int, production, colors, pips, is_tutor)
        card_classes[cls.name].append(display_name)

    # Deck-wide colored-pip demand (quantity-weighted), computed once at
    # classification -- never inside the Monte Carlo loop. Consumed by the
    # Color Screen section below.
    pip_demand: dict[str, int] = {}
    for qty, name in slots:
        for color in slot_by_key[name.lower()].pips:
            pip_demand[color] = pip_demand.get(color, 0) + qty

    deck: list[_Slot] = []
    for qty, name in slots:
        deck.extend([slot_by_key[name.lower()]] * qty)

    stats = _run_simulation(
        deck,
        iterations=iterations,
        seed=seed,
        min_lands=min_lands,
        max_lands=max_lands,
        count_mdfc_lands=count_mdfc_lands,
        keep_rule=keep_rule,
        free_mulligan=free_mulligan,
        gas_cmc_threshold=gas_cmc_threshold,
        commander_colors=parsed_colors,
        tutor_aware=tutor_aware,
    )

    lines = [
        "# Opening Hand Simulation",
        "",
        f"**Simulating a {deck_size}-card library** (commander already excluded)",
        f"**Iterations:** {iterations}" + (f"  **Seed:** {seed}" if seed is not None else ""),
        f"**Land Range:** {min_lands}-{max_lands} effective lands to keep",
        f"**Keep Rule:** {keep_rule}  **Free Mulligan:** {free_mulligan}",
        "",
        "## Keep Rates",
        "",
        "| Hand Size | Keep % |",
        "|---|---|",
    ]
    for size in (7, 6, 5, 4):
        label = f"{size} (forced)" if size == 4 else str(size)
        lines.append(f"| {label} | {stats['keep_pct_by_hand_size'][size]:.1%} |")
    lines.append("")
    lines.append("| Mulligans | Keep % |")
    lines.append("|---|---|")
    for mull_label in ("0", "1", "2", "3plus"):
        lines.append(f"| {mull_label} | {stats['keep_pct_by_mulligans'][mull_label]:.1%} |")
    lines.append("")
    mull_reasons = stats["mull_reasons"]
    mull_line = (
        f"- Mulled hands: {mull_reasons['screw']} screw, {mull_reasons['flood']} flood, "
        f"{mull_reasons['no_gas']} no gas"
    )
    if parsed_colors:
        mull_line += f", {mull_reasons['colors']} off-color"
    lines.append(mull_line)
    if keep_rule == "playability":
        lines.append(
            "- Playability rule: a hand is kept based on a 3-turn goldfish of the hand "
            "alone (development speed, flood, and castable gas)."
        )
    else:
        lines.append("- Legacy lands_v1 rule: a hand is kept based on effective land count only.")
    lines.append(
        "- Commander free mulligan: the first mulligan redraws a full 7 without bottoming a card."
        if free_mulligan
        else "- No free mulligan: every mulligan bottoms a card, including the first."
    )
    lines.append("")

    keep_first_deal_pct = stats["keep_pct_by_mulligans"]["0"]
    keep_via_free_mulligan_pct = stats["keep_pct_by_mulligans"]["1"]
    lines.append("## Mulligan Transparency")
    lines.append("")
    lines.append(
        f"- Kept on the first 7 (0 mulligans, the deck's true consistency): "
        f"{keep_first_deal_pct:.1%}"
    )
    lines.append(
        f"- Saved by the Commander free mulligan (kept after exactly 1 mulligan, "
        f"still 7 cards): {keep_via_free_mulligan_pct:.1%}"
    )
    lines.append(f"- Total kept at 7 cards: {stats['keep_pct_by_hand_size'][7]:.1%}")
    lines.append(
        "*`free_mulligan` is already simulated above: the first line is what the deck "
        "does on its own, the second is what the free redraw is covering for it.*"
    )
    lines.append("")

    lines.append("## Kept Hand Land Distribution")
    lines.append("")
    if keep_rule == "lands_v1":
        lines.append(
            "*Only lands (not rocks/dorks) count toward the keep decision -- v1 limitation.*"
        )
    else:
        lines.append(
            "*Land distribution of kept hands under the playability rule "
            "(development/flood/gas, not lands alone).*"
        )
    lines.append("")
    lines.append("| Effective Lands | % of Kept Hands |")
    lines.append("|---|---|")
    for lands in sorted(stats["kept_land_distribution"]):
        lines.append(f"| {lands:g} | {stats['kept_land_distribution'][lands]:.1%} |")
    lines.append("")

    lines.append("## Mana by Turn")
    lines.append("")
    lines.append(
        "*Spendable mana is the NET pool left after casting this turn's mana rocks/"
        "dorks/ramp spells, not the raw total produced. A turn spent playing a rock "
        'can read "off curve" even though it is setting up bigger turns ahead.*'
    )
    lines.append("")
    lines.append("| Turn | Avg Spendable Mana | On Curve % |")
    lines.append("|---|---|---|")
    for turn in range(1, 6):
        avg = stats["avg_spendable_mana_by_turn"][turn]
        on_curve = stats["on_curve_pct_by_turn"][turn]
        lines.append(f"| {turn} | {avg:.2f} | {on_curve:.1%} |")
    lines.append("")

    lines.append("## Ramp")
    lines.append("")
    lines.append(f"- Mana source cast by turn 2: {stats['ramp_by_turn2_pct']:.1%}")
    lines.append("")

    hand_composition = stats["hand_composition"]
    lines.append("## Hand Composition")
    lines.append("")
    lines.append("*Averages and rates over KEPT hands only -- does not affect keep/bottom rules.*")
    lines.append("")
    lines.append(f"- Avg lands per hand: {hand_composition['avg_lands']:.2f}")
    lines.append(f"- Avg rocks/dorks per hand: {hand_composition['avg_rocks_dorks']:.2f}")
    if tutor_aware:
        lines.append(f"- Avg tutors per hand: {hand_composition['avg_tutors']:.2f}")
    else:
        lines.append("- Avg tutors per hand: not tracked (enable `tutor_aware`)")
    lines.append(
        f"- Avg gas (cmc <= {gas_cmc_threshold}) per hand: {hand_composition['avg_gas']:.2f}"
    )
    lines.append(f"- Avg cmc 6+ cards per hand: {hand_composition['avg_cmc6plus']:.2f}")
    lines.append(f"- Hands with 0 castable gas: {hand_composition['pct_no_gas']:.1%}")
    lines.append(f"- Hands with 2+ cmc-6+ cards: {hand_composition['pct_2plus_cmc6']:.1%}")
    color_source_pct = hand_composition["color_source_pct"]
    if color_source_pct:
        color_line = ", ".join(f"{c} {pct:.1%}" for c, pct in color_source_pct.items())
        lines.append(f"- Hands sourcing each deck color (>= 1 source): {color_line}")
    lines.append(
        "*Reading tip: the most underrepresented category above is the one to "
        "reinforce in the list.*"
    )
    lines.append("")

    color_screen: dict[str, object] = {}
    if parsed_colors:
        color_mull_pct = mull_reasons["colors"] / iterations
        color_screen = {
            "commander_colors": sorted(parsed_colors),
            "color_mull_pct": color_mull_pct,
        }
        lines.append("## Color Screen")
        lines.append("")
        lines.append(f"- Commander colors: {', '.join(sorted(parsed_colors))}")
        lines.append(f"- Hands mulliganed for missing a color: {color_mull_pct:.1%}")
        if pip_demand:
            demand_str = ", ".join(f"{c}:{pip_demand[c]}" for c in sorted(pip_demand))
            lines.append(f"- Deck color demand (colored pips across spells): {demand_str}")
        lines.append(
            "*A hand is screened out when it cannot source every commander color "
            "(playability rule only).*"
        )
        lines.append("")

    if tutors:
        lines.append("## Tutors")
        lines.append("")
        lines.append(f"- Opening hand holds a tutor: {stats['tutor_in_hand_pct']:.1%}")
        lines.append("")
        lines.append("| Tutor | Finds | To | Speed | Targets |")
        lines.append("|---|---|---|---|---|")
        for tutor in sorted(tutors, key=lambda t: t.name):
            lines.append(
                f"| {tutor.name} | {tutor.target_constraint} | {tutor.destination} "
                f"| {tutor.speed} | {tutor.target_count} |"
            )
        lines.append("")

    lines.append("## Detected Card Classes")
    lines.append("")
    for cls in (
        _CardClass.LAND,
        _CardClass.MDFC_LAND,
        _CardClass.ROCK,
        _CardClass.DORK,
        _CardClass.RAMP_SPELL,
    ):
        names = card_classes[cls.name]
        if names:
            lines.append(f"- **{cls.value}** ({len(names)}): {', '.join(sorted(names))}")
    lines.append("")

    if unresolved:
        lines.append("## Warnings")
        lines.append("")
        lines.append(f"- {len(unresolved)} card(s) could not be resolved: {', '.join(unresolved)}")
        lines.append("")

    log.info("simulate_opening_hands.complete", deck_size=deck_size, iterations=iterations)
    data: dict[str, object] = {
        "iterations": iterations,
        "seed": seed,
        "deck_size": deck_size,
        **stats,
        "keep_first_deal_pct": keep_first_deal_pct,
        "keep_via_free_mulligan_pct": keep_via_free_mulligan_pct,
        "card_classes": card_classes,
        "tutors": [t._asdict() for t in tutors],
        "color_screen": color_screen,
        "unresolved": unresolved,
        "params": {
            "min_lands": min_lands,
            "max_lands": max_lands,
            "count_mdfc_lands": count_mdfc_lands,
            "keep_rule": keep_rule,
            "free_mulligan": free_mulligan,
            "gas_cmc_threshold": gas_cmc_threshold,
            "extra_mana_sources": sorted(extra_set),
            "exclude_cards": sorted(exclude_set),
            "commander_colors": sorted(parsed_colors),
            "tutor_aware": tutor_aware,
        },
    }
    return WorkflowResult(markdown="\n".join(lines), data=data)
