"""Derive how often an ability triggers, and on what.

Card data says what an ability does. It does not say how many times it does it, and
that is where a deck evaluation goes wrong: "Whenever a Ninja you control deals combat
damage" fires once per Ninja that connects, while "Whenever one or more Ninja deal
combat damage" fires once, full stop. Same shape, different card entirely.

This is a reading of the oracle text, not a rules engine. It surfaces the distinction
so a human or a model has to confront it; it does not resolve the ability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mtg_mcp_server.types import Card

__all__ = ["Trigger", "derive_trigger"]

# Ordered: the first condition that matches wins, so the specific forms come first.
# "combat damage to a player" must be tested before the bare "combat damage".
_CONDITIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("combat_damage_to_player", re.compile(r"combat damage to (?:a |target )?player", re.I)),
    ("combat_damage", re.compile(r"combat damage", re.I)),
    ("attacks", re.compile(r"\battacks?\b", re.I)),
    ("blocks", re.compile(r"\bblocks?\b", re.I)),
    ("enters", re.compile(r"\benters\b", re.I)),
    ("dies", re.compile(r"\bdies\b", re.I)),
    ("cast", re.compile(r"\bcasts?\b", re.I)),
    ("phase_step", re.compile(r"at the beginning of|at end of", re.I)),
)

_TRIGGER_WORD = re.compile(r"\b(whenever|when|at the beginning of|at end of)\b", re.I)

# "one or more X ... deal" is the batched form: one trigger for the whole combat.
_BATCHED = re.compile(r"\bone or more\b", re.I)

# "a Ninja you control", "another creature", "each other" — a class of sources, so the
# ability triggers separately for each member that meets the condition.
_MULTI_SOURCE = re.compile(
    r"\bwhenever\s+(?:a|an|another|each|one)\b(?!\s+or\s+more)",
    re.I,
)

_SELF_SOURCE = re.compile(r"\bwhenever\s+(?:this|~|it)\b", re.I)

# Same class-of-sources idea as _MULTI_SOURCE, but entry triggers use "when" as often
# as "whenever", so the combat regex misses them.
_ENTRY_MULTI = re.compile(
    r"\b(?:when|whenever)\s+(?:a|an|another|each|one)\b(?!\s+or\s+more)",
    re.I,
)

_ATTACKS = re.compile(r"\battacks?\b", re.I)
_PER_OPPONENT = re.compile(r"\beach opponent\b", re.I)
_PER_TURN = re.compile(r"at the beginning of|at end of", re.I)
_ON_ENTRY = re.compile(r"\b(?:when|whenever)\b[^.]*\benters\b", re.I)

_ZERO_POWER_CAVEAT = (
    "Combat damage trigger: a creature with 0 power deals no combat damage and never "
    "triggers this, whatever its types (510.1a). Blocked attackers deal damage to the "
    "blocker, not the player."
)


@dataclass(frozen=True)
class Trigger:
    """How often an ability fires, and what fires it.

    ``scope`` is the field that changes an evaluation:

    - ``per_source``   — once per qualifying permanent (multiplies with attackers)
    - ``per_combat``   — once per combat, however many creatures qualify
    - ``per_turn``     — a phase or step trigger
    - ``per_opponent`` — once per opponent
    - ``on_entry``     — an enters-the-battlefield trigger
    - ``static``       — no trigger at all
    """

    scope: str
    condition: str | None = None
    sources: str | None = None  # "self" | "class" when the scope is per_source
    notes: str | None = None
    other_scopes: tuple[str, ...] = ()  # scopes of the card's OTHER trigger clauses


# Widest reach first. A card carrying several triggers is reported by its widest one,
# because that is the one that changes an evaluation; the rest go to `other_scopes`.
_SCOPE_RANK = {
    "per_source": 0,
    "per_opponent": 1,
    "per_combat": 2,
    "per_turn": 3,
    "on_entry": 4,
    "static": 5,
}

_CLAUSE_SPLIT = re.compile(r"\n|(?<=[.!])\s+")


def _condition_of(oracle: str) -> str | None:
    for name, pattern in _CONDITIONS:
        if pattern.search(oracle):
            return name
    return None


def _derive_one(clause: str) -> Trigger | None:
    """Derive the trigger of a SINGLE clause, or None if the clause has no trigger."""
    if not _TRIGGER_WORD.search(clause):
        return None

    condition = _condition_of(clause)
    notes = (
        _ZERO_POWER_CAVEAT if condition in ("combat_damage", "combat_damage_to_player") else None
    )

    # "enters OR attacks" is one clause carrying two conditions. Reporting only the
    # entry half hides a trigger that fires every combat.
    if _ON_ENTRY.search(clause):
        if _ATTACKS.search(clause):
            return Trigger(
                scope="per_combat",
                condition="enters_or_attacks",
                sources="self",
                notes=notes,
            )
        # An entry trigger naming a CLASS of permanents fires once per permanent that
        # enters. Impact Tremors doubles with every token; ranking it `on_entry` — the
        # second-narrowest scope — dropped it out of the trigger-reach section
        # entirely, which is the exact blindness this module exists to remove.
        if _BATCHED.search(clause):
            return Trigger(scope="per_combat", condition="enters", notes=notes)
        if _ENTRY_MULTI.search(clause):
            return Trigger(scope="per_source", condition="enters", sources="class", notes=notes)
        return Trigger(scope="on_entry", condition="enters", notes=notes)
    if _PER_TURN.search(clause):
        return Trigger(scope="per_turn", condition=condition or "phase_step", notes=notes)
    if _PER_OPPONENT.search(clause):
        return Trigger(scope="per_opponent", condition=condition, notes=notes)

    # The distinction this module exists for. "one or more" batches every qualifying
    # source into a single trigger; "a Ninja you control" does not.
    if _BATCHED.search(clause):
        return Trigger(scope="per_combat", condition=condition, notes=notes)
    if _SELF_SOURCE.search(clause):
        return Trigger(scope="per_source", condition=condition, sources="self", notes=notes)
    if _MULTI_SOURCE.search(clause):
        return Trigger(scope="per_source", condition=condition, sources="class", notes=notes)

    return Trigger(scope="per_combat", condition=condition, notes=notes)


def _triggers_of(oracle: str) -> list[Trigger]:
    return [
        t
        for t in (_derive_one(c) for c in _CLAUSE_SPLIT.split(oracle) if c.strip())
        if t is not None
    ]


def derive_trigger(card: Card) -> Trigger:
    """Read a card's oracle text and report how its ability triggers.

    The oracle is split into clauses FIRST. A card carrying both "At the beginning of
    your upkeep..." and "Whenever a Ninja you control deals combat damage..." has two
    separate abilities; judging the whole text at once let whichever pattern matched
    first win, and silently buried the multiplying trigger — the exact distinction
    this module exists to surface. The widest-reaching clause is reported; the others
    are listed in ``other_scopes`` so nothing is hidden.
    """
    oracle = card.oracle_text or ""
    if not oracle:
        return Trigger(scope="static")

    triggers = _triggers_of(oracle)

    # Older templating names the card instead of saying "this creature" ("Whenever
    # Kalonian Hydra attacks"). Normalising it lets the self-source rule see it.
    if card.name and card.name in oracle and not any(t.sources for t in triggers):
        renamed = _triggers_of(oracle.replace(card.name, "this creature"))
        if renamed:
            triggers = renamed

    if not triggers:
        return Trigger(scope="static")

    widest = min(triggers, key=lambda t: _SCOPE_RANK.get(t.scope, 99))
    others = tuple(sorted({t.scope for t in triggers} - {widest.scope}))
    return Trigger(
        scope=widest.scope,
        condition=widest.condition,
        sources=widest.sources,
        notes=widest.notes,
        other_scopes=others,
    )
