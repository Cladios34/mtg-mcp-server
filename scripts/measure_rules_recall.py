"""Measure rules retrieval recall against the annotated question set.

Runs every question in tests/fixtures/rules/annotated_questions.json through
rules_scenario against the real Comprehensive Rules corpus, and reports where
the expected rules landed.

    uv run python scripts/measure_rules_recall.py
    uv run python scripts/measure_rules_recall.py --verbose   # per-question detail

The split by phrasing is the point. Questions naming a glossary term retrieve
well; questions phrased the way a player actually speaks are the ones that fail.
A single average hides that.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

QUESTIONS = REPO / "tests" / "fixtures" / "rules" / "annotated_questions.json"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="print per-question detail")
    args = parser.parse_args()

    from mtg_mcp_server.config import Settings
    from mtg_mcp_server.services.rules import RulesService
    from mtg_mcp_server.workflows.rules import rules_scenario

    payload = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = payload["questions"]

    service = RulesService(rules_url=Settings().rules_url, refresh_hours=168)
    await service.ensure_loaded()

    # Guard: an expected rule that does not exist would silently score as a miss
    # forever, and the tuning it motivated would chase a rule that is not there.
    missing: list[tuple[str, str]] = []
    for q in questions:
        for number in q["expected_rules"]:
            if await service.lookup_by_number(number) is None:
                missing.append((q["id"], number))
    if missing:
        print("ANNOTATION ERROR — these expected rules do not exist in the corpus:")
        for qid, number in missing:
            print(f"  {qid}: {number}")
        return 1

    rows = []
    for q in questions:
        result = await rules_scenario(q["question_en"], rules=service)
        ranked = [r["number"] for r in result.data["rules"]]
        expected = q["expected_rules"]
        ranks = {e: (ranked.index(e) + 1 if e in ranked else None) for e in expected}
        found = [r for r in ranks.values() if r is not None]
        rows.append(
            {
                "id": q["id"],
                "family": q["family"],
                "phrasing": q["phrasing"],
                "expected": expected,
                "ranks": ranks,
                "best": min(found) if found else None,
                "all_top10": bool(found) and all(r is not None and r <= 10 for r in ranks.values()),
                "returned": len(ranked),
                "size": len(result.markdown),
            }
        )

    def summarize(label: str, subset: list[dict]) -> None:
        if not subset:
            return
        n = len(subset)
        at1 = sum(1 for r in subset if r["best"] == 1)
        at5 = sum(1 for r in subset if r["best"] and r["best"] <= 5)
        at10 = sum(1 for r in subset if r["best"] and r["best"] <= 10)
        full = sum(1 for r in subset if r["all_top10"])
        never = sum(1 for r in subset if r["best"] is None)
        mrr = sum(1 / r["best"] for r in subset if r["best"]) / n
        print(
            f"{label:<22} n={n:<3} "
            f"@1={at1 / n:5.0%}  @5={at5 / n:5.0%}  @10={at10 / n:5.0%}  "
            f"full@10={full / n:5.0%}  absent={never / n:5.0%}  MRR={mrr:.3f}"
        )

    print(f"\ncorpus: {payload['corpus']}   questions: {len(rows)}\n")
    summarize("ALL", rows)
    print()
    for phrasing in ("named", "plain"):
        summarize(f"phrasing={phrasing}", [r for r in rows if r["phrasing"] == phrasing])
    print()
    for family in sorted({r["family"] for r in rows}):
        summarize(f"  {family}", [r for r in rows if r["family"] == family])

    misses = [r for r in rows if not r["best"] or r["best"] > 5]
    print(f"\n--- outside top 5 ({len(misses)}/{len(rows)}) ---")
    for r in sorted(misses, key=lambda r: (r["best"] is not None, r["best"] or 0), reverse=True):
        detail = ", ".join(f"{k}@{v if v else 'absent'}" for k, v in r["ranks"].items())
        print(f"{r['id']:<12} {r['phrasing']:<6} {detail}   (returned {r['returned']})")

    if args.verbose:
        print("\n--- every question ---")
        for r in rows:
            detail = ", ".join(f"{k}@{v if v else 'absent'}" for k, v in r["ranks"].items())
            print(f"{r['id']:<12} {r['phrasing']:<6} {detail:<40} {r['size'] // 1000}KB")

    sizes = [r["size"] for r in rows]
    print(
        f"\nresponse size: median {sorted(sizes)[len(sizes) // 2] // 1000} KB, max {max(sizes) // 1000} KB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
