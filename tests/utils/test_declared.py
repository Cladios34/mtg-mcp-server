"""Tests for declared-category measurement.

Regression origin (2026-07-27): the owner declared 13 cheap creatures, the list had 13,
the work shipped 11, and nothing connected the two — while the probability of opening
one fell from 63.91% to 57.36%.
"""

from __future__ import annotations

from typing import ClassVar

from mtg_mcp_server.types import Card
from mtg_mcp_server.utils.declared import evaluate_categories, parse_filter


def _card(name: str, *, cmc: float = 1.0, type_line: str = "Creature — Ninja", **kwargs) -> Card:
    return Card.model_validate(
        {"id": name.lower(), "name": name, "cmc": cmc, "type_line": type_line} | kwargs
    )


DECK = [
    _card("Changeling Outcast", cmc=1.0),
    _card("Mothdust Changeling", cmc=1.0),
    _card("Faerie Seer", cmc=1.0),
    _card("Sol Ring", cmc=1.0, type_line="Artifact"),
    _card("Yuriko, the Tiger's Shadow", cmc=3.0),
]


class TestFilterParsing:
    def test_mana_value_upper_bound(self) -> None:
        parsed = parse_filter("mv<=1")
        assert parsed.mv_lte == 1

    def test_strict_less_than_is_converted(self) -> None:
        assert parse_filter("mv<2").mv_lte == 1

    def test_type_term(self) -> None:
        assert parse_filter("t:creature").types == ("creature",)

    def test_combined_terms(self) -> None:
        parsed = parse_filter("mv<=1 t:creature")
        assert parsed.mv_lte == 1
        assert parsed.types == ("creature",)

    def test_quoted_oracle_term(self) -> None:
        assert parse_filter('o:"deals combat damage"').text == ("deals combat damage",)

    def test_unparsed_terms_are_reported_not_dropped(self) -> None:
        # A filter that silently ignores half its input produces a count that looks
        # authoritative and is not.
        assert parse_filter("mv<=1 something_weird").unparsed == ("something_weird",)


class TestOrOperator:
    """Regression origin (2026-07-30) : ``t:angel or t:demon or t:dragon`` sur un deck
    qui contenait 16 de ces créatures a rendu actual=0, les jetons ``or`` avalés dans
    unparsed_terms : un compte faux mais plausible. La règle attendue : ``or`` fait
    l'UNION des clauses, le ET implicite reste entre termes juxtaposés, et ``or``
    lie moins fort que la juxtaposition (``t:a or t:b mv<=3`` = t:a OU (t:b ET mv<=3)).
    """

    TRIBES: ClassVar[list[Card]] = [
        _card("Serra Angel", type_line="Creature — Angel", cmc=5.0),
        _card("Razaketh", type_line="Creature — Demon", cmc=8.0),
        _card("Shivan Dragon", type_line="Creature — Dragon", cmc=6.0),
        _card("Sol Ring", type_line="Artifact", cmc=1.0),
    ]

    def test_or_makes_a_union_and_leaves_nothing_unparsed(self) -> None:
        parsed = parse_filter("t:angel or t:demon or t:dragon")
        assert parsed.unparsed == ()
        matched = [c.name for c in self.TRIBES if parsed.matches(c)]
        assert matched == ["Serra Angel", "Razaketh", "Shivan Dragon"]

    def test_or_binds_looser_than_juxtaposition(self) -> None:
        # t:angel or t:dragon mv<=3  ==  t:angel OU (t:dragon ET mv<=3)
        parsed = parse_filter("t:angel or t:dragon mv<=3")
        assert parsed.unparsed == ()
        assert parsed.matches(_card("Serra Angel", type_line="Creature — Angel", cmc=5.0))
        assert not parsed.matches(_card("Shivan Dragon", type_line="Creature — Dragon", cmc=6.0))
        assert parsed.matches(_card("Dragon Hatchling", type_line="Creature — Dragon", cmc=2.0))

    def test_quoted_or_is_text_not_operator(self) -> None:
        parsed = parse_filter('o:"destroy or exile"')
        assert parsed.text == ("destroy or exile",)
        assert parsed.unparsed == ()

    def test_dangling_or_is_reported_not_swallowed(self) -> None:
        # Un « or » sans clause de chaque côté ne doit jamais disparaître en silence.
        assert "or" in parse_filter("t:angel or").unparsed
        assert "or" in parse_filter("or t:angel").unparsed

    def test_union_count_through_evaluate_categories(self) -> None:
        # La forme exacte de l'incident : 3 types déclarés, compte attendu exact.
        result = evaluate_categories(
            [{"name": "tribes", "filter": "t:angel or t:demon or t:dragon", "expected": 3}],
            self.TRIBES,
        )[0]
        assert result.actual == 3
        assert result.drift == 0
        assert result.drifted is False
        assert result.to_dict()["unparsed_terms"] == []


class TestDriftDetection:
    def test_matching_declaration_has_no_drift(self) -> None:
        result = evaluate_categories(
            [{"name": "cheap creatures", "filter": "mv<=1 t:creature", "expected": 3}], DECK
        )[0]
        assert result.actual == 3
        assert result.drift == 0
        assert result.drifted is False

    def test_shrunk_category_is_flagged(self) -> None:
        # The exact shape of the incident: declared 13, list holds 11.
        result = evaluate_categories(
            [{"name": "cheap creatures", "filter": "mv<=1 t:creature", "expected": 5}], DECK
        )[0]
        assert result.actual == 3
        assert result.drift == -2
        assert result.drifted is True

    def test_no_expectation_means_no_drift_claim(self) -> None:
        result = evaluate_categories([{"name": "cheap", "filter": "mv<=1 t:creature"}], DECK)[0]
        assert result.expected is None
        assert result.drift is None
        assert result.drifted is False

    def test_matched_cards_are_named(self) -> None:
        result = evaluate_categories([{"name": "cheap", "filter": "mv<=1 t:creature"}], DECK)[0]
        assert "Sol Ring" not in result.cards
        assert "Changeling Outcast" in result.cards

    def test_explicit_card_list(self) -> None:
        result = evaluate_categories(
            [{"name": "the three", "cards": ["Faerie Seer", "Sol Ring", "Not In Deck"]}], DECK
        )[0]
        assert result.actual == 2
        assert result.drift == -1

    def test_several_categories_at_once(self) -> None:
        results = evaluate_categories(
            [
                {"name": "cheap creatures", "filter": "mv<=1 t:creature", "expected": 3},
                {"name": "artifacts", "filter": "t:artifact", "expected": 2},
            ],
            DECK,
        )
        assert [r.drift for r in results] == [0, -1]

    def test_to_dict_round_trip(self) -> None:
        result = evaluate_categories(
            [{"name": "cheap", "filter": "mv<=1 t:creature", "expected": 5}], DECK
        )[0].to_dict()
        assert result["expected"] == 5
        assert result["actual"] == 3
        assert result["drift"] == -2
        assert result["drifted"] is True
