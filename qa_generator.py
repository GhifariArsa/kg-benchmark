"""
Stage 2: QA generator (graph-agnostic).

Reads a knowledge-graph JSON, samples stratified chains by (subcategory, depth),
generates dev-language questions via LLM, proposes accepted/rejected answer
sets, applies a deterministic graph-topology filter, and emits a question bank
with first-class coverage stats.

Output schema is locked in memory/qa_design.md.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


# === Configuration ===

SUBCATEGORIES_BY_GRAPH = {
    "call_graph": ["dependency", "reachability", "entry_point", "responsibility"],
    # data_flow / control_flow / contract / definition deferred until those
    # extractors exist. `definition` is also low-value (see qa_design.md).
}

SUBCATEGORY_TO_L1 = {
    "dependency": "where",
    "reachability": "how",
    "entry_point": "where",
    "responsibility": "where",
    "definition": "what",
    "data_flow": "where",
    "control_flow": "how",
    "contract": "what",
}

# Target depth distribution (≥70% at depth ≥2 — Sillito/LaToza pain-point bias).
# Profiles indexed by max depth; larger repos warrant deeper chains.
DEPTH_PROFILES = {
    3: {1: 0.30, 2: 0.45, 3: 0.25},
    4: {1: 0.20, 2: 0.40, 3: 0.30, 4: 0.10},
    5: {1: 0.15, 2: 0.30, 3: 0.30, 4: 0.15, 5: 0.10},
    6: {1: 0.10, 2: 0.25, 3: 0.25, 4: 0.20, 5: 0.12, 6: 0.08},
}

MODEL = "qwen3.6"
ACCEPTED_HOP_LIMIT = 2  # accepted_answers must be within N undirected hops of canonical
LOCAL_SUBGRAPH_HOPS = 2


# === Graph utilities ===


def load_graph(path):
    with open(path) as f:
        return json.load(f)


def build_adjacency(graph):
    """Forward (caller→callees), reverse (callee→callers), undirected adj."""
    fwd = defaultdict(set)
    rev = defaultdict(set)
    und = defaultdict(set)
    for e in graph["edges"]:
        c, t = e["caller"], e["callee"]
        fwd[c].add(t)
        rev[t].add(c)
        und[c].add(t)
        und[t].add(c)
    return (
        {k: list(v) for k, v in fwd.items()},
        {k: list(v) for k, v in rev.items()},
        {k: set(v) for k, v in und.items()},
    )


def enumerate_chains(adj, start, depth, max_chains=20):
    """Simple paths of exactly `depth` edges from `start` (depth+1 nodes)."""
    results = []

    def dfs(node, path):
        if len(results) >= max_chains:
            return
        if len(path) - 1 == depth:
            results.append(path[:])
            return
        for nxt in adj.get(node, ()):
            if nxt in path:  # cycle break: simple paths only
                continue
            path.append(nxt)
            dfs(nxt, path)
            path.pop()

    dfs(start, [start])
    return results


def hop_distance(und, a, b, limit):
    """Undirected BFS distance, capped at `limit`. Returns float('inf') if > limit."""
    if a == b:
        return 0
    seen = {a}
    frontier = {a}
    for d in range(1, limit + 1):
        nxt = set()
        for n in frontier:
            for m in und.get(n, ()):
                if m == b:
                    return d
                if m not in seen:
                    seen.add(m)
                    nxt.add(m)
        frontier = nxt
    return float("inf")


def local_subgraph_nodes(und, center, chain, hops):
    """All nodes within `hops` undirected hops of `center`, plus chain nodes."""
    seen = {center}
    frontier = {center}
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            for m in und.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    nxt.add(m)
        frontier = nxt
    seen.update(chain)
    return seen


# === Sampling ===


def chain_target(chain, subcategory):
    """Which node in the chain is the canonical answer."""
    if subcategory == "entry_point":
        return chain[
            0
        ]  # entry_point asks "what triggers chain[-1]"; answer is upstream
    return chain[-1]


def chain_primary_edge(chain, subcategory):
    """The single edge the question is 'about' — incident to the canonical."""
    if len(chain) < 2:
        return None
    if subcategory == "entry_point":
        return [chain[0], chain[1]]
    return [chain[-2], chain[-1]]


def stratified_plan(n_questions, graph_view, depth_distribution):
    """List of (subcategory, depth) pairs totalling n_questions."""
    subcats = SUBCATEGORIES_BY_GRAPH[graph_view]
    n_per = n_questions // len(subcats)
    remainder = n_questions - n_per * len(subcats)

    plan = []
    for i, sc in enumerate(subcats):
        n = n_per + (1 if i < remainder else 0)
        for d, frac in depth_distribution.items():
            plan += [(sc, d)] * int(round(n * frac))
        # pad to n
        while sum(1 for s, _ in plan if s == sc) < n:
            plan.append((sc, 2))

    random.shuffle(plan)
    return plan


def sample_chain(adj, candidates, depth, max_attempts=50):
    """Random simple path of `depth` edges. None if can't find one in attempts."""
    pool = list(candidates)
    random.shuffle(pool)
    for start in pool[:max_attempts]:
        chains = enumerate_chains(adj, start, depth, max_chains=10)
        if chains:
            return random.choice(chains)
    return None


# === LLM ===

GENERATE_SYSTEM = """You generate code-comprehension benchmark questions from a knowledge graph.

Inputs you receive:
- graph_view: the kind of graph (e.g. call_graph)
- subcategory: the kind of question to generate
- chain: a path of node ids through the graph
- target_answer: the node id that MUST be the canonical answer

Subcategory phrasing guidance:
- dependency: ask which function is called/used (forward direction). Answer = end of chain.
- reachability: ask what eventually gets invoked when starting from the chain head. Answer = end of chain.
- entry_point: ask which public/upstream API triggers the END of the chain. Answer = START of chain.
- responsibility: ask which function is responsible for / handles a behavior described semantically. Answer = end of chain.

CRITICAL — anti-ambiguity rule:
The question must point UNAMBIGUOUSLY to `target_answer`. A developer reading ONLY the question (without seeing the chain or the graph) must be able to identify `target_answer` as the unique correct answer.
- Do NOT phrase the question in a way that more naturally points to a parent class, a public-API alias, or a more general method than `target_answer`.
- If the most natural reading of the question would lead a developer to a different node than `target_answer`, the question is WRONG — rewrite it.
- Anchor the question with specific behavior, intent, or context that distinguishes `target_answer` from its neighbors.

Style rules:
- Phrase by intent / responsibility / locality, NOT graph-language.
- Bad: "What does X call after Y?"
- Good: "When the scheduler decides a job is due, which method actually executes it?"
- The agent will see ONLY the question (not the chain). Make the question stand on its own.
- Refer to nodes by their short name when natural, never invent identifiers.

Return ONLY valid JSON with keys:
- question: string
- canonical_answer: string (must equal target_answer EXACTLY)
"""

PROPOSE_SYSTEM = """You propose alternate accepted answers and close-but-wrong answers for a code-comprehension question.

Inputs:
- question
- canonical_answer (the most precise correct answer; a graph node id)
- chain: the call path the question was derived from
- nearby: list of node ids within 2 hops of the canonical

Choose ONLY from `nearby`. Do not invent ids.

Guidance for accepted_answers — be INCLUSIVE:
- INCLUDE nodes immediately adjacent to the canonical on the chain (predecessor or successor). They sit on the same reasoning path; an agent that names them has demonstrated correct understanding.
- INCLUDE near-synonyms: public-API aliases, parent classes when the question doesn't pin to a specific child, helper methods that do the same job.
- An agent that names any of these has answered the question correctly enough.

Guidance for rejected_but_close:
- Nodes from `nearby` that an agent might plausibly but INCORRECTLY name — wrong but on the right topic. These are the most informative wrong answers for error analysis.

Return ONLY valid JSON with keys:
- accepted_answers: list of node ids (excluding canonical) that would also be defensible. May be empty.
- rejected_but_close: list of nearby ids that would be wrong-but-tempting. May be empty.
"""


def llm_generate_question(chain, subcategory, graph_view, target_answer):
    user = (
        f"graph_view: {graph_view}\n"
        f"subcategory: {subcategory}\n"
        f"chain: {' -> '.join(chain)}\n"
        f"target_answer: {target_answer}"
    )
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": GENERATE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(resp.choices[0].message.content or "{}")


def llm_propose_answers(question, canonical, chain, nearby):
    user = (
        f"question: {question}\n"
        f"canonical_answer: {canonical}\n"
        f"chain: {' -> '.join(chain)}\n"
        f"nearby: {json.dumps(sorted(nearby))}"
    )
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROPOSE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(resp.choices[0].message.content or "{}")


def filter_answers(proposed, canonical, all_node_ids, local_node_ids, und):
    """Deterministic filter: valid graph nodes, ≤N undirected hops from canonical."""
    accepted = {canonical}
    rejected_close = set()

    for cand in proposed.get("accepted_answers", []) or []:
        if cand not in all_node_ids or cand == canonical:
            continue
        if hop_distance(und, canonical, cand, ACCEPTED_HOP_LIMIT) > ACCEPTED_HOP_LIMIT:
            if cand in local_node_ids:
                rejected_close.add(cand)
            continue
        accepted.add(cand)

    for cand in proposed.get("rejected_but_close", []) or []:
        if cand in all_node_ids and cand not in accepted:
            rejected_close.add(cand)

    return sorted(accepted), sorted(rejected_close)


# === Coverage ===


class Coverage:
    def __init__(self, graph):
        self.all_nodes = {n["id"] for n in graph["nodes"]}
        self.all_edges = {(e["caller"], e["callee"]) for e in graph["edges"]}
        self.node_hits = defaultdict(int)
        self.edge_hits = defaultdict(int)
        self.by_depth = defaultdict(int)
        self.by_subcategory = defaultdict(int)

    def record(self, chain, primary_edge, depth, subcategory):
        for n in chain:
            self.node_hits[n] += 1
        if primary_edge:
            self.edge_hits[tuple(primary_edge)] += 1
        self.by_depth[depth] += 1
        self.by_subcategory[subcategory] += 1

    def report(self):
        nodes_covered = sum(1 for n in self.all_nodes if self.node_hits[n] > 0)
        edges_covered = sum(1 for e in self.all_edges if self.edge_hits[e] > 0)
        uncovered = [list(e) for e in self.all_edges if self.edge_hits[e] == 0]
        n_nodes, n_edges = max(1, len(self.all_nodes)), max(1, len(self.all_edges))
        return {
            "edges_covered": f"{edges_covered}/{len(self.all_edges)} ({100 * edges_covered / n_edges:.0f}%)",
            "nodes_covered": f"{nodes_covered}/{len(self.all_nodes)} ({100 * nodes_covered / n_nodes:.0f}%)",
            "by_depth": dict(sorted(self.by_depth.items())),
            "by_subcategory": dict(self.by_subcategory),
            "uncovered_edges": uncovered[:50],
            "uncovered_count": len(uncovered),
        }


# === Generation pipeline ===


def is_noisy_target(node_id):
    """Skip canonicals that are structural noise (e.g. __init__ from class instantiation)."""
    return node_id.endswith(".__init__")


def generate_one(graph, fwd, und, all_node_ids, subcategory, depth, graph_view, idx):
    # Retry sampling if canonical lands on a noisy node (e.g. *.__init__).
    chain = None
    for _ in range(8):
        candidate = sample_chain(fwd, all_node_ids, depth)
        if candidate is None:
            return None
        candidate_target = chain_target(candidate, subcategory)
        if is_noisy_target(candidate_target):
            continue
        chain = candidate
        break
    if chain is None:
        return None

    canonical = chain_target(chain, subcategory)
    primary_edge = chain_primary_edge(chain, subcategory)

    try:
        q = llm_generate_question(chain, subcategory, graph_view, canonical)
    except Exception as e:
        print(f"  ✗ generate failed: {e}", file=sys.stderr)
        return None

    # Force canonical to the sampled target, in case the LLM drifted.
    if q.get("canonical_answer") not in all_node_ids:
        q["canonical_answer"] = canonical
    canonical = q["canonical_answer"]

    nearby = local_subgraph_nodes(und, canonical, chain, LOCAL_SUBGRAPH_HOPS)

    try:
        proposed = llm_propose_answers(q["question"], canonical, chain, nearby)
    except Exception as e:
        print(f"  ✗ propose failed: {e}", file=sys.stderr)
        proposed = {"accepted_answers": [], "rejected_but_close": []}

    accepted, rejected_close = filter_answers(
        proposed, canonical, all_node_ids, nearby, und
    )

    record = {
        "id": f"{graph.get('repo', 'unknown')}__{graph_view}__q{idx:04d}",
        "repo": graph.get("repo"),
        "graph_views": [graph_view],
        "category": SUBCATEGORY_TO_L1.get(subcategory, "where"),  # deterministic
        "subcategory": subcategory,
        "depth": depth,
        "question": q["question"],
        "canonical_answer": canonical,
        "accepted_answers": accepted,
        "rejected_but_close": rejected_close,
        "provenance": {
            "chain": chain,
            "primary_edges": [primary_edge] if primary_edge else [],
        },
    }
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA bank from a knowledge graph."
    )
    parser.add_argument("graph_path", help="Path to graph JSON")
    parser.add_argument("-n", "--num-questions", type=int, default=50)
    parser.add_argument("--graph-view", default="call_graph")
    parser.add_argument("--max-depth", type=int, default=4, choices=sorted(DEPTH_PROFILES))
    parser.add_argument("-o", "--output", default="qa_output/question_bank.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    depth_distribution = DEPTH_PROFILES[args.max_depth]

    graph = load_graph(args.graph_path)
    fwd, _rev, und = build_adjacency(graph)
    all_node_ids = {n["id"] for n in graph["nodes"]}

    if args.graph_view not in SUBCATEGORIES_BY_GRAPH:
        print(f"Unknown graph_view: {args.graph_view}", file=sys.stderr)
        sys.exit(1)

    plan = stratified_plan(args.num_questions, args.graph_view, depth_distribution)
    print(
        f"Generating {len(plan)} questions for graph view '{args.graph_view}' "
        f"on '{graph.get('repo', '?')}' ({len(all_node_ids)} nodes, "
        f"{len(graph['edges'])} edges)"
    )

    coverage = Coverage(graph)
    questions = []
    idx = 0
    for subcategory, depth in plan:
        rec = generate_one(
            graph, fwd, und, all_node_ids, subcategory, depth, args.graph_view, idx
        )
        if rec is None:
            continue
        questions.append(rec)
        coverage.record(
            rec["provenance"]["chain"],
            rec["provenance"]["primary_edges"][0]
            if rec["provenance"]["primary_edges"]
            else None,
            depth,
            subcategory,
        )
        idx += 1
        print(f"  ✓ [{subcategory:15s} d={depth}] {rec['question'][:90]}")

    output = {
        "source": args.graph_path,
        "graph_view": args.graph_view,
        "repo": graph.get("repo"),
        "num_questions": len(questions),
        "coverage": coverage.report(),
        "questions": questions,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    cov = coverage.report()
    print(f"\nSaved {len(questions)} questions → {args.output}")
    print(f"  Edge coverage      : {cov['edges_covered']}")
    print(f"  Node coverage      : {cov['nodes_covered']}")
    print(f"  By depth           : {cov['by_depth']}")
    print(f"  By subcategory     : {cov['by_subcategory']}")
    print(f"  Uncovered edges    : {cov['uncovered_count']} (first 50 in output)")


if __name__ == "__main__":
    main()
