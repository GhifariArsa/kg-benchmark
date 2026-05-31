"""
Stage 3b: Scorer.

Reads agent_runs.json + question_bank.json. For each run:
1. Extract a node id from the agent's free-form answer (LLM-as-judge, narrow).
2. Deterministic match: extracted in question.accepted_answers?
3. Classify error using rejected_but_close.

Outputs scored_results.json with per-run grades and aggregated metrics
broken down by agent × depth × subcategory × L1 category.
"""

import argparse
import json
import os
from collections import defaultdict

from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"


EXTRACT_SYSTEM = """You extract the node id that an agent named as the answer to a code-comprehension question.

You receive:
- valid_node_ids: list of node ids from the codebase
- agent_answer: free-form text from the agent

Return ONLY valid JSON with one key:
- extracted: a single id from valid_node_ids that the agent named as the answer, or null if no match.

Rules:
- The id must come from valid_node_ids exactly. Do not invent.
- If multiple ids appear, pick the one the agent presents as the FINAL answer (typically after "Answer:" or similar).
- A short-name match (e.g. agent says "_run_job") may correspond to a qualified id ("schedule.Scheduler._run_job") — prefer the qualified id from valid_node_ids.
- If no valid id matches, return null.
"""


def extract_node_id(agent_answer, valid_ids):
    if not agent_answer:
        return None
    valid_str = json.dumps(sorted(valid_ids))
    if len(valid_str) > 8000:
        valid_str = valid_str[:8000] + "...]"
    user = (
        f"valid_node_ids: {valid_str}\n"
        f"agent_answer: {agent_answer[-4000:]}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content).get("extracted")
    except Exception:
        return None


def classify_error(extracted, accepted, rejected_close):
    if extracted in accepted:
        return None
    if extracted is None:
        return "extraction_failed"
    if extracted in rejected_close:
        return "close"
    return "wildly_wrong"


def acc(items):
    if not items:
        return None
    return round(100 * sum(1 for x in items if x.get("correct")) / len(items), 1)


def aggregate(items):
    depths = sorted({x["depth"] for x in items if x.get("depth") is not None})
    subcats = sorted({x["subcategory"] for x in items if x.get("subcategory")})
    cats = sorted({x["category"] for x in items if x.get("category")})
    return {
        "n": len(items),
        "accuracy": acc(items),
        "by_depth": {d: acc([x for x in items if x.get("depth") == d]) for d in depths},
        "by_subcategory": {
            sc: acc([x for x in items if x.get("subcategory") == sc]) for sc in subcats
        },
        "by_l1_category": {
            c: acc([x for x in items if x.get("category") == c]) for c in cats
        },
        "errors": {
            cls: sum(1 for x in items if x.get("error_class") == cls)
            for cls in ["close", "wildly_wrong", "extraction_failed", "no_run"]
        },
        "avg_elapsed_s": round(
            sum(x.get("elapsed_s") or 0 for x in items) / max(1, len(items)), 2
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Score agent runs against the question bank.")
    parser.add_argument("runs_path", help="Path to agent_runs.json")
    parser.add_argument("bank_path", help="Path to question_bank.json")
    parser.add_argument("-o", "--output", default="qa_output/scored_results.json")
    args = parser.parse_args()

    with open(args.runs_path) as f:
        runs_data = json.load(f)
    with open(args.bank_path) as f:
        bank = json.load(f)

    qmap = {q["id"]: q for q in bank["questions"]}

    valid_ids = set()
    for q in bank["questions"]:
        valid_ids.add(q["canonical_answer"])
        valid_ids.update(q["accepted_answers"])
        valid_ids.update(q["rejected_but_close"])
    valid_ids = sorted(valid_ids)

    print(f"Scoring {len(runs_data['runs'])} runs against {len(qmap)} questions...")

    scored = []
    for run in runs_data["runs"]:
        q = qmap.get(run["question_id"])
        if q is None:
            continue
        if not run.get("completed"):
            scored.append(
                {
                    **run,
                    "extracted": None,
                    "correct": False,
                    "error_class": "no_run",
                    "depth": q["depth"],
                    "subcategory": q["subcategory"],
                    "category": q["category"],
                }
            )
            continue

        extracted = extract_node_id(run.get("agent_answer") or "", valid_ids)
        correct = extracted in q["accepted_answers"]
        err = classify_error(
            extracted, set(q["accepted_answers"]), set(q["rejected_but_close"])
        )
        scored.append(
            {
                **run,
                "extracted": extracted,
                "correct": correct,
                "error_class": err,
                "depth": q["depth"],
                "subcategory": q["subcategory"],
                "category": q["category"],
            }
        )
        mark = "✓" if correct else ("~" if err == "close" else "✗")
        print(
            f"  {mark} [{run['agent']:12s}] {run['question_id']}  "
            f"extracted={extracted}  canonical={q['canonical_answer']}"
        )

    by_agent = defaultdict(list)
    for s in scored:
        by_agent[s["agent"]].append(s)

    summary = {agent: aggregate(items) for agent, items in by_agent.items()}

    output = {
        "source_runs": args.runs_path,
        "source_bank": args.bank_path,
        "summary": summary,
        "runs": scored,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved → {args.output}\n")
    for agent, s in summary.items():
        print(f"{agent}: {s['accuracy']}%  ({s['n']} questions, avg {s['avg_elapsed_s']}s)")
        print(f"  by depth        : {s['by_depth']}")
        print(f"  by subcategory  : {s['by_subcategory']}")
        print(f"  by L1 category  : {s['by_l1_category']}")
        print(f"  errors          : {s['errors']}")
        print()


if __name__ == "__main__":
    main()
