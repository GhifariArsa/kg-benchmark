"""
Stage 2: Q&A generator.
Takes the call graph JSON from extractor.py and generates a question bank
using an LLM. Only uses resolved edges (internal calls).
"""

import json
import sys
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """You are a benchmark question generator for code repositories.
Given a function call edge from a Python codebase, generate a natural language 
question and its ground truth answer that tests whether a coding agent understands 
the codebase structure.

Rules:
- Questions must require knowing the codebase graph, not just reading one line
- Questions should sound like what a developer would actually ask
- Avoid trivial questions (e.g. "what parameters does X take?")
- Answer must be the exact qualified function name from the graph
- Return only valid JSON, no preamble

Output format:
{
  "question": "...",
  "answer": "...",
  "difficulty": "single-hop"
}"""


def generate_question(caller: str, callee: str) -> dict | None:
    prompt = f"""Codebase: Python `schedule` library (job scheduling)
Caller: {caller}
Callee: {callee}

Generate a question that tests whether an agent knows that {caller} calls {callee}."""

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    text = response.choices[0].message.content.strip()
    # strip possible markdown fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def generate_multihop_question(chain: list[str]) -> dict | None:
    """Generate a question requiring traversal of a 2-hop call chain."""
    path = " → ".join(chain)
    prompt = f"""Codebase: Python `schedule` library (job scheduling)
Call chain: {path}

Generate a question that requires knowing this entire call chain to answer correctly.
For example: "If I call {chain[0]}, what function ultimately handles X?"
The answer should be {chain[-1]}."""

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    result = json.loads(text.strip())
    result["difficulty"] = "multi-hop"
    result["chain"] = chain
    return result


def build_chains(edges: list[dict]) -> list[list[str]]:
    """Find 2-hop chains: A→B→C from resolved edges."""
    # build adjacency
    adj = {}
    for e in edges:
        if e["callee_resolved"]:
            adj.setdefault(e["caller"], []).append(e["callee_resolved"])

    chains = []
    for a, bs in adj.items():
        for b in bs:
            if b in adj:
                for c in adj[b]:
                    chains.append([a, b, c])
    return chains


def main(graph_path: str):
    graph = json.loads(open(graph_path).read())

    resolved_edges = [e for e in graph["edges"] if e["callee_resolved"]]

    # deduplicate caller→callee pairs
    seen = set()
    unique_edges = []
    for e in resolved_edges:
        key = (e["caller"], e["callee_resolved"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    print(f"Generating questions for {len(unique_edges)} unique resolved edges...")

    question_bank = []

    # --- Single-hop questions ---
    for e in unique_edges:
        try:
            q = generate_question(e["caller"], e["callee_resolved"])
            q["edge"] = f"{e['caller']} → {e['callee_resolved']}"
            q["difficulty"] = "single-hop"
            question_bank.append(q)
            print(f"  ✓ {e['caller']} → {e['callee_resolved']}")
        except Exception as ex:
            print(f"  ✗ Failed: {e['caller']} → {e['callee_resolved']}: {ex}")

    # --- Multi-hop questions (sample first 5 chains to keep scope small) ---
    chains = build_chains(unique_edges)
    print(f"\nGenerating multi-hop questions for {min(5, len(chains))} chains...")
    for chain in chains[:5]:
        try:
            q = generate_multihop_question(chain)
            question_bank.append(q)
            print(f"  ✓ {' → '.join(chain)}")
        except Exception as ex:
            print(f"  ✗ Failed: {chain}: {ex}")

    output = {
        "source": graph_path,
        "num_questions": len(question_bank),
        "questions": question_bank,
    }

    out_path = "question_bank.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(question_bank)} questions to {out_path}")
    return output


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "call_graph.json"
    main(path)
