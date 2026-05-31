"""
Batch QA bank generator.

Iterates over every *_call_graph.json in --graph-dir, sizes each repo by edge
count, picks (n_questions, max_depth), and runs qa_generator.py for each.
Outputs to qa_output/{repo}_question_bank.json.
"""

import argparse
import glob
import json
import os
import subprocess
import sys


# (min_edges, n_questions, max_depth)
SIZE_TIERS = [
    (0, 200, 4),  # tiny  (e.g. schedule, tenacity, loguru)
    (200, 250, 5),  # mid   (e.g. typer)
    (500, 300, 6),  # large (e.g. httpx, requests, instructor)
]


def pick_params(n_edges):
    for min_e, n, depth in reversed(SIZE_TIERS):
        if n_edges >= min_e:
            return n, depth
    return SIZE_TIERS[0][1], SIZE_TIERS[0][2]


def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate QA banks for all repos."
    )
    parser.add_argument(
        "--graph-dir", default="output", help="Where *_call_graph.json files live."
    )
    parser.add_argument("--out-dir", default="qa_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip repos whose question bank already exists.",
    )
    args = parser.parse_args()

    graph_files = sorted(glob.glob(os.path.join(args.graph_dir, "*_call_graph.json")))
    if not graph_files:
        sys.exit(f"No *_call_graph.json found under {args.graph_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Found {len(graph_files)} graph files in {args.graph_dir}\n")

    for gf in graph_files:
        with open(gf) as f:
            g = json.load(f)
        repo = g.get("repo") or os.path.basename(gf).replace("_call_graph.json", "")
        n_edges = len(g.get("edges", []))
        n_nodes = len(g.get("nodes", []))
        n_questions, max_depth = pick_params(n_edges)

        out_path = os.path.join(args.out_dir, f"{repo}_question_bank.json")
        if args.skip_existing and os.path.exists(out_path):
            print(f"⊘ {repo:15s}  bank exists, skipping")
            continue

        print(
            f"▶ {repo:15s}  nodes={n_nodes:4d}  edges={n_edges:5d}  "
            f"→ n={n_questions}  max_depth={max_depth}"
        )

        cmd = [
            sys.executable,
            "qa_generator.py",
            gf,
            "-n",
            str(n_questions),
            "--max-depth",
            str(max_depth),
            "-o",
            out_path,
            "--seed",
            str(args.seed),
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  ✗ failed (exit {rc})", file=sys.stderr)
        print()


if __name__ == "__main__":
    main()
