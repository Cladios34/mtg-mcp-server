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
    "MAX_KEYWORD_LENGTH",
    "AlternativeCost",
    "ParsedCost",
    "ReductionResult",
    "TypeMatch",
    "alternative_cast_cost",
    "apply_generic_reduction",
    "card_keywords",
    "carries_keyword",
    "creature_types",
    "has_creature_type",
    "keyword_activation_cost",
    "mana_value",
    "normalize_keyword",
    "parse_mana_cost",
    "reduction_amount",
]

# The longest keyword Scryfall lists is "More Than Meets the Eye", at 23 characters.
# 64 leaves room for anything Wizards prints next while keeping the bound meaningful.
#
# GOTCHA(2026-07-29): the cost pattern cache is bounded in entry COUNT but not in
# entry SIZE, and compiling the regex is linear in the keyword's length ON THE
# SERVER'S SINGLE EVENT LOOP. Measured: 819 ms at 100 KB, 3.3 s at 400 KB, minutes
# at a few MB. One call was enough to freeze the server for every client, and the
# cache then held 256 of them. The keyword is a tool argument on a public,
# unauthenticated endpoint, so it is checked before anything is compiled.
MAX_KEYWORD_LENGTH = 64

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

_REDUCTION_AMOUNT = re.compile(r"\{(\d)\}\s+less", re.IGNORECASE)

_CHOSEN_TYPE = re.compile(r"\bthe chosen (?:creature )?type\b", re.IGNORECASE)

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


def _reject_oversized_keyword(keyword: str) -> None:
    """Refuse a keyword no Magic card could carry, before it costs anything.

    Raises:
        ValueError: If ``keyword`` is longer than :data:`MAX_KEYWORD_LENGTH`.
    """
    if len(keyword) > MAX_KEYWORD_LENGTH:
        raise ValueError(
            f"keyword too long: {len(keyword)} characters, limit is {MAX_KEYWORD_LENGTH}. "
            "The longest Magic keyword is 23 characters."
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
    _reject_oversized_keyword(keyword)
    wanted = normalize_keyword(keyword)
    if wanted in card_keywords(card):
        return True
    return keyword_activation_cost(card, wanted) is not None


def reduction_clause(card: Card) -> str | None:
    """The sentence that carries the cost reduction, verbatim.

    A reducer's amount is only half the story: "Spells you cast of the chosen type
    cost {1} less" reduces nothing at all if the chosen type is not the deck's. This
    module cannot resolve a choice made at resolution time, so it hands back the
    clause and lets the reader judge the scope instead of asserting one.
    """
    for sentence in re.split(r"(?<=[.!])\s+|\n", card.oracle_text or ""):
        if any(phrase in sentence.lower() for phrase in _REDUCTION_PHRASES):
            return sentence.strip()
    return None


def reduction_depends_on_choice(card: Card) -> bool:
    """Whether the reduction's scope is decided by a choice, not by the card text."""
    return bool(_CHOSEN_TYPE.search(reduction_clause(card) or ""))


def has_reduction_clause(card: Card) -> bool:
    """Whether the oracle reduces costs at all, however unquotable the amount.

    ``reduction_amount`` returns None both for "no clause" and for a clause it refuses
    to put a number on. Callers that report the difference to a human need to tell the
    two apart: "it does not reduce costs" is a very different sentence from "it does,
    by an amount that depends on the board".
    """
    return any(phrase in (card.oracle_text or "").lower() for phrase in _REDUCTION_PHRASES)


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
    amounts = {int(found) for found in _REDUCTION_AMOUNT.findall(oracle)}
    if len(amounts) > 1:
        # Two different reductions in one oracle ("{2} less to cast", "{1} less to
        # activate"). Returning the larger is a precise, plausible, wrong number for
        # whichever cost the caller meant — the same failure this function already
        # refuses for variable reductions, so it gets the same answer.
        return None
    if amounts:
        return amounts.pop()
    return 1


@lru_cache(maxsize=256)
def _keyword_cost_pattern(keyword: str) -> re.Pattern[str]:
    """Compile (and cache) the activation-cost pattern for a keyword.

    Bounded on purpose. ``keyword`` reaches here from a tool argument a client
    controls freely (``cost_reduction_check(keyword=...)``, and the ``kw:`` term
    of a category filter), so an unbounded module-level dict would be a memory
    leak any caller could grow at will.

    "Basic landcycling" and "Commander ninjutsu" are distinct keywords that carry the
    same activation shape as their base form, so both prefixes are optional here.
    """
    _reject_oversized_keyword(keyword)
    return re.compile(
        rf"(?:commander\s+|basic\s+)?{re.escape(keyword)}\s*((?:\{{[^}}]+\}})+)",
        re.IGNORECASE,
    )


def keyword_activation_cost(card: Card, keyword: str) -> str | None:
    """Extract the mana cost printed right after ``keyword`` in the oracle text.

    Raises:
        ValueError: If ``keyword`` is longer than :data:`MAX_KEYWORD_LENGTH`.
    """
    _reject_oversized_keyword(keyword)
    oracle = card.oracle_text
    if not oracle:
        return None
    match = _keyword_cost_pattern(keyword).search(oracle)
    return match.group(1) if match else None


def _symbol_value(symbol: str) -> int:
    """Mana value of a single cost symbol.

    ``{X}`` is zero on the stack (rule 202.3b). A hybrid is worth the greater of
    its halves (rule 202.3f), so ``{2/U}`` is 2: counting it as 1 would make a
    card look cheaper than it can ever be cast for.
    """
    upper = symbol.upper()
    if upper == "X":
        return 0
    if "/" in upper:
        halves = [int(part) for part in upper.split("/") if part.isdigit()]
        return max(halves) if halves else 1
    return 1


def mana_value(mana_cost: str | None) -> int:
    """Mana value of a printed cost string, as an integer."""
    parsed = parse_mana_cost(mana_cost)
    return parsed.generic + sum(_symbol_value(s) for s in parsed.symbols)


@dataclass(frozen=True)
class AlternativeCost:
    """A cheaper way to cast a card from hand than paying its printed cost."""

    value: int
    keyword: str
    cost: str


# Keywords that let a card be cast FROM HAND for a different mana cost, with no
# prerequisite this module cannot see. Deliberately short: Escape, Flashback,
# Disturb and Jump-start cast from the graveyard; Madness needs a discard;
# Miracle needs the draw window; Foretell splits payment across two turns;
# Bestow and Overload cost MORE. None of those answer "what can this hand cast
# on curve", so counting them would trade one wrong number for another.
#
# GOTCHA(2026-07-29): Impending prints as "Impending 4—{2}{W}{W}": a counter and
# a dash sit between the keyword and the cost, so the plain keyword-then-cost
# pattern misses it. It is a real carrier of this bug class: Overlord of the
# Mistmoors is a {5}{W}{W} card castable for 4. Note it enters as a NON-creature
# until its last time counter is removed, so it is a castable spell on curve but
# not a body on curve; the simulation measures the former.
_ALTERNATIVE_CAST_KEYWORDS = ("Warp", "Evoke", "Impending")

# Keyword, then an optional reminder counter and dash, then the mana cost.
# The dash class carries the three Scryfall actually prints (em, en, hyphen);
# ruff flags the literals as ambiguous, so they are spelled by code point.
_ALTERNATIVE_COST_RE = r"{keyword}(?:\s+\d+)?\s*[—–-]?\s*((?:\{{[^}}]+\}})+)"  # noqa: RUF001


@lru_cache(maxsize=32)
def _alternative_cost_pattern(keyword: str) -> re.Pattern[str]:
    """Compile (and cache) the alternative-cost pattern for a keyword.

    Separate from :func:`_keyword_cost_pattern` on purpose: that one serves
    activation costs reached from a client-controlled argument, and widening it
    to swallow "Impending 4—" would change what every caller of it matches.
    Bounded cache, though this one only ever sees the constants above.
    """
    return re.compile(_ALTERNATIVE_COST_RE.format(keyword=re.escape(keyword)), re.IGNORECASE)


def alternative_cast_cost(card: Card) -> AlternativeCost | None:
    """The cheapest alternative cost this card can be cast from hand for.

    Returns ``None`` when the card has no such cost, when the alternative is not
    payable in mana, or when it is not actually cheaper than the printed cost.

    Two independent checks must agree before a discount is reported: the keyword
    must appear in Scryfall's ``keywords`` field, AND a mana cost must be printed
    immediately after it in the oracle text. Either alone produces false
    positives: Mutalith Vortex Beast carries the keyword "Warp Vortex" (the name
    of a triggered ability), and reminder text names keywords it does not grant.
    Declining costs nothing: the caller falls back to the printed mana value,
    which is what it used before this function existed.
    """
    printed = round(card.cmc)
    keywords = card_keywords(card)
    best: AlternativeCost | None = None

    for keyword in _ALTERNATIVE_CAST_KEYWORDS:
        if keyword not in keywords:
            continue
        match = _alternative_cost_pattern(keyword).search(card.oracle_text or "")
        if match is None:
            # A non-mana alternative cost (Grief's "Evoke-Exile a black card").
            # Whether it is payable depends on the rest of the hand.
            continue
        cost = match.group(1)
        value = mana_value(cost)
        if value >= printed:
            continue
        if best is None or value < best.value:
            best = AlternativeCost(value=value, keyword=keyword, cost=cost)

    return best


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
