"""Mechanical reasoning about mana costs, keywords, and creature types.

Everything here replaces a judgement call that was made — and made wrong — during a
real deck audit. The rules are mechanical, so the answers should be computed rather
than recalled:

- Rule 601.2f: a generic cost reduction reduces the generic part of a cost and
  nothing else. Coloured pips survive it.
- Rule 702.73a: a changeling is every creature type, everywhere, always. A tribal
  count that omits changelings is wrong by construction.
- A permanent that becomes a typed creature under a condition still belongs in the
  count — with its condition attached, never silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mtg_mcp_server.types import Card

__all__ = [
    "ParsedCost",
    "ReductionResult",
    "TypeMatch",
    "apply_generic_reduction",
    "card_keywords",
    "carries_keyword",
    "creature_types",
    "has_creature_type",
    "keyword_activation_cost",
    "normalize_keyword",
    "parse_mana_cost",
    "reduction_amount",
]

_REDUCTION_PHRASES = ("less to cast", "less to activate", "costs less", "less to play")

# Keywords shared by half of Magic. They describe nothing about how a deck operates,
# so they are never a deck's signature mechanic. Canonical here because two callers
# (mechanic_map, audit_bundle) need the same list, and two copies would drift apart
# the first time a new evergreen keyword is printed.
EVERGREEN_KEYWORDS = frozenset(
    {
        "flying",
        "first strike",
        "double strike",
        "deathtouch",
        "haste",
        "hexproof",
        "indestructible",
        "lifelink",
        "menace",
        "reach",
        "trample",
        "vigilance",
        "defender",
        "flash",
        "ward",
        "protection",
        "equip",
        "enchant",
    }
)

_SYMBOL_RE = re.compile(r"\{([^}]+)\}")

# "costs {1} less to cast for each artifact you control" — the amount is not fixed,
# so no single number describes it honestly.
_VARIABLE_REDUCTION = re.compile(r"less to (?:cast|activate|play)[^.]*\bfor each\b", re.IGNORECASE)

# Scryfall type lines use an em dash. The double hyphen is the ASCII form that
# shows up in hand-typed lists.
_TYPE_SEPARATORS = ("—", "--")

_CHANGELING_ORACLE = re.compile(r"\bis every creature type\b", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!])\s+|\n")


@dataclass(frozen=True)
class ParsedCost:
    """A mana cost split into what a generic reduction can touch and what it cannot.

    ``generic`` counts ONLY plain numeric symbols. ``{X}``, ``{C}``, and hybrids like
    ``{2/U}`` are symbols: they look generic-ish and are not.
    """

    generic: int = 0
    symbols: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render back to Scryfall notation. A cost with nothing left is ``{0}``."""
        parts = []
        if self.generic > 0:
            parts.append(f"{{{self.generic}}}")
        parts.extend(f"{{{s}}}" for s in self.symbols)
        return "".join(parts) if parts else "{0}"


@dataclass(frozen=True)
class ReductionResult:
    """Outcome of applying a generic cost reduction to one cost."""

    cost: str
    reduced: bool
    result: str
    reason: str


@dataclass(frozen=True)
class TypeMatch:
    """Whether a card has a creature type, and on what grounds.

    ``via`` is the part that matters downstream: a printed type, a changeling, and a
    conditional creature are all "yes", but they are not the same "yes" and a deck
    report that flattens them is misleading.
    """

    matches: bool
    via: str | None = None  # "printed" | "changeling" | "conditional"
    condition: str | None = None


def parse_mana_cost(mana_cost: str | None) -> ParsedCost:
    """Split a mana cost into its reducible generic part and its fixed symbols."""
    if not mana_cost:
        return ParsedCost()

    generic = 0
    symbols: list[str] = []
    for raw in _SYMBOL_RE.findall(mana_cost):
        if raw.isdigit():
            generic += int(raw)
        else:
            symbols.append(raw)
    return ParsedCost(generic=generic, symbols=symbols)


def apply_generic_reduction(mana_cost: str | None, amount: int) -> ReductionResult:
    """Apply a generic cost reduction of ``amount`` to ``mana_cost`` (rule 601.2f).

    A reduction removes generic mana only. ``{U}{B}`` reduced by ``{1}`` is still
    ``{U}{B}`` — the claim that a pair of reducers takes such a cost to ``{0}`` is the
    single most expensive mistake this module exists to prevent.
    """
    cost = mana_cost or ""
    parsed = parse_mana_cost(cost)
    rendered = parsed.render()

    if amount <= 0:
        return ReductionResult(cost=rendered, reduced=False, result=rendered, reason="no reduction")

    if parsed.generic == 0:
        return ReductionResult(
            cost=rendered,
            reduced=False,
            result=rendered,
            reason=(
                "no generic component in the PRINTED cost — a generic reduction cannot "
                "reduce coloured, colorless or hybrid symbols (601.2f). Note {X}: once a "
                "value is chosen, X counts as generic mana and IS reducible, which this "
                "function cannot resolve from the printed cost alone"
            ),
        )

    shaved = min(amount, parsed.generic)
    reduced = ParsedCost(generic=parsed.generic - shaved, symbols=parsed.symbols)
    return ReductionResult(
        cost=rendered,
        reduced=True,
        result=reduced.render(),
        reason=f"generic part reduced by {shaved}",
    )


def normalize_keyword(keyword: str) -> str:
    """Normalise a keyword for comparison across cards.

    "Commander ninjutsu" folds into "Ninjutsu". They are distinct keywords in the
    rules, but a deck's package includes both, and splitting them fragments exactly
    the count a mechanic map exists to produce.
    """
    stripped = keyword.strip()
    without_prefix = (
        stripped[len("commander ") :] if stripped.lower().startswith("commander ") else stripped
    )
    if not without_prefix:
        return ""
    return without_prefix[0].upper() + without_prefix[1:]


def card_keywords(card: Card) -> set[str]:
    """The card's keywords, normalised via :func:`normalize_keyword`."""
    return {normalize_keyword(k) for k in card.keywords if k.strip()}


def carries_keyword(card: Card, keyword: str) -> bool:
    """Does this card actually have ``keyword``?

    Checked two ways on purpose. Scryfall's ``keywords`` field is the clean signal but
    it is not always populated, and dropping a carrier silently reproduces the very
    undercount these tools exist to prevent. The second check requires the keyword to
    be followed by an activation cost, so a card that merely TALKS about the mechanic
    ("Ninjutsu abilities cost {1} less") is not counted as carrying it.
    """
    wanted = normalize_keyword(keyword)
    if wanted in card_keywords(card):
        return True
    return keyword_activation_cost(card, wanted) is not None


def reduction_amount(card: Card) -> int | None:
    """How much generic mana this card shaves off costs, or None if it shaves none.

    Returns None for a VARIABLE reduction ("costs {1} less for each artifact you
    control"). Reporting such a card as a flat -{1} is the same failure mode as the
    fixed-reduction errors this module exists to prevent: a number that is precise,
    plausible, and wrong. A caller that cannot get a fixed amount must say so rather
    than quote one.
    """
    oracle = (card.oracle_text or "").lower()
    if not any(phrase in oracle for phrase in _REDUCTION_PHRASES):
        return None
    if _VARIABLE_REDUCTION.search(oracle):
        return None
    for amount in range(9, 0, -1):
        if f"{{{amount}}} less" in oracle:
            return amount
    return 1


@lru_cache(maxsize=256)
def _keyword_cost_pattern(keyword: str) -> re.Pattern[str]:
    """Compile (and cache) the activation-cost pattern for a keyword.

    Bounded on purpose. ``keyword`` reaches here from a tool argument a client
    controls freely (``cost_reduction_check(keyword=...)``), so an unbounded
    module-level dict would be a memory leak any caller could grow at will.

    "Basic landcycling" and "Commander ninjutsu" are distinct keywords that carry the
    same activation shape as their base form, so both prefixes are optional here.
    """
    return re.compile(
        rf"(?:commander\s+|basic\s+)?{re.escape(keyword)}\s*((?:\{{[^}}]+\}})+)",
        re.IGNORECASE,
    )


def keyword_activation_cost(card: Card, keyword: str) -> str | None:
    """Extract the mana cost printed right after ``keyword`` in the oracle text."""
    oracle = card.oracle_text
    if not oracle:
        return None
    match = _keyword_cost_pattern(keyword).search(oracle)
    return match.group(1) if match else None


def creature_types(card: Card) -> set[str]:
    """Printed creature subtypes from the type line.

    Subtypes of non-creature permanents come back too (the type line does not say
    which half a subtype belongs to); callers that care check the card type first.
    """
    line = card.type_line or ""
    for separator in _TYPE_SEPARATORS:
        if separator in line:
            return set(line.split(separator, 1)[1].split())
    return set()


def _is_changeling(card: Card) -> bool:
    """Rule 702.73a — changelings are every creature type, in every zone."""
    if any(k.lower() == "changeling" for k in card.keywords):
        return True
    oracle = card.oracle_text or ""
    return "changeling" in oracle.lower() or bool(_CHANGELING_ORACLE.search(oracle))


def _conditional_type_clause(card: Card, creature_type: str) -> str | None:
    """Find the sentence granting ``creature_type`` under a condition.

    Kaito, Bane of Nightmares is the case that matters: a planeswalker that is a Ninja
    creature during your turn while it has loyalty counters. It belongs in a tribal
    count, but only with the condition spelled out.
    """
    oracle = card.oracle_text or ""
    needle = creature_type.lower()
    # Cheap gate before the split: the clause has to name both the type and the word
    # "creature", so a text missing either cannot match. Sweeping the bulk data for a
    # rare tribe used to run the split on every card's oracle — 138ms of blocking work
    # over ~30k cards, against 5ms for the plain substring test it replaced.
    lowered = oracle.lower()
    if not oracle or needle not in lowered or "creature" not in lowered:
        return None
    for sentence in _SENTENCE_SPLIT.split(oracle):
        lowered = sentence.lower()
        if needle in lowered and "creature" in lowered:
            return sentence.strip()
    return None


def has_creature_type(card: Card, creature_type: str) -> TypeMatch:
    """Does ``card`` have ``creature_type``, and by what mechanism?

    Checked in order of certainty: printed type, then changeling, then a conditional
    grant in the oracle text.

    A creature subtype belongs to a CREATURE. "Kindred Sorcery — Otter" is a sorcery
    that shares a tribe's name; counting it as a member inflates every tribal total
    with cards that never hit the battlefield as creatures.
    """
    wanted = creature_type.lower()
    type_line = (card.type_line or "").lower()
    is_creature = "creature" in type_line

    if is_creature and any(t.lower() == wanted for t in creature_types(card)):
        return TypeMatch(matches=True, via="printed")

    if is_creature and _is_changeling(card):
        return TypeMatch(
            matches=True,
            via="changeling",
            condition="changeling — every creature type, in every zone (702.73a)",
        )

    # A conditional grant is the one route open to a non-creature: a planeswalker
    # that becomes a creature under a condition still belongs in the count.
    clause = _conditional_type_clause(card, creature_type)
    if clause is not None:
        return TypeMatch(matches=True, via="conditional", condition=clause)

    return TypeMatch(matches=False)
