"""
extractor.py — Call graph extractor for coding agent benchmark
Uses pyan3 for static analysis. Works on any Python repo.
Output: call_graph.json with nodes, edges, and metadata.
"""

import os
import sys
import json
import argparse
import glob
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pyan.analyzer import CallGraphVisitor
from pyan.node import Flavor


GITHUB_URL_RE = re.compile(
    r"^(https?://github\.com/|git@github\.com:|git://github\.com/)"
)


def is_github_url(s: str) -> bool:
    return bool(GITHUB_URL_RE.match(s)) or s.endswith(".git")


def repo_name_from_url(url: str) -> str:
    base = url.rstrip("/").rsplit("/", 1)[-1]
    return base[:-4] if base.endswith(".git") else base


@contextmanager
def cloned_repo(url: str):
    """Shallow-clone url into a tempdir; yield path; clean up on exit."""
    with tempfile.TemporaryDirectory(prefix="kg_extractor_") as tmp:
        dest = os.path.join(tmp, repo_name_from_url(url))
        print(f"Cloning {url} → {dest} (shallow)...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, dest],
                check=True,
                stdout=sys.stderr,
                stderr=sys.stderr,
            )
        except subprocess.CalledProcessError as e:
            raise SystemExit(f"git clone failed: {e}")
        yield dest


# Flavors we consider real, defined code entities
VALID_FLAVORS = {Flavor.FUNCTION, Flavor.METHOD, Flavor.CLASS}


def find_python_files(repo_path: str) -> list[str]:
    """Recursively find all .py files in a repo, skipping common noise dirs."""
    skip_dirs = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "dist",
        "build",
        ".eggs",
        "*.egg-info",
        "site-packages",
        "tests",
        "test",
        "docs",
    }

    py_files = []
    for root, dirs, files in os.walk(repo_path):
        # Prune skip dirs in-place
        dirs[:] = [
            d for d in dirs if d not in skip_dirs and not d.endswith(".egg-info")
        ]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))

    return sorted(py_files)


def node_id(node) -> str:
    """Canonical string ID for a node: namespace.name"""
    if node.namespace:
        return f"{node.namespace}.{node.name}"
    return node.name


def is_valid_node(node) -> bool:
    """Filter to only defined, real code entities (functions, methods, classes)."""
    return (
        node.defined
        and node.flavor in VALID_FLAVORS
        and node.namespace is not None  # exclude top-level module nodes
        and "^^^" not in node.name  # exclude internal pyan placeholders
    )


def build_call_graph(py_files: list[str], repo_path: str) -> dict:
    """
    Run pyan on the given files and return a structured call graph dict.
    """
    if not py_files:
        raise ValueError("No Python files found in repo.")

    visitor = CallGraphVisitor(py_files)

    # --- Build node registry ---
    nodes = {}
    for name, nodelist in visitor.nodes.items():
        for node in nodelist:
            if is_valid_node(node):
                nid = node_id(node)
                if nid not in nodes:
                    # Compute relative file path for readability
                    rel_file = (
                        os.path.relpath(node.filename, repo_path)
                        if node.filename
                        else None
                    )
                    nodes[nid] = {
                        "id": nid,
                        "name": node.name,
                        "namespace": node.namespace,
                        "type": node.flavor.name.lower(),  # function / method / class
                        "file": rel_file,
                    }

    # --- Build edge list ---
    edges = []
    seen_edges = set()

    for caller_node, callee_set in visitor.uses_edges.items():
        if not is_valid_node(caller_node):
            continue
        caller_id = node_id(caller_node)
        if caller_id not in nodes:
            continue

        for callee_node in callee_set:
            if not is_valid_node(callee_node):
                continue
            callee_id = node_id(callee_node)
            if callee_id not in nodes:
                continue

            edge_key = (caller_id, callee_id)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            edges.append(
                {
                    "caller": caller_id,
                    "callee": callee_id,
                }
            )

    # --- Stats ---
    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "functions": sum(1 for n in nodes.values() if n["type"] == "function"),
        "methods": sum(1 for n in nodes.values() if n["type"] == "method"),
        "classes": sum(1 for n in nodes.values() if n["type"] == "class"),
        "files_analyzed": len(py_files),
    }

    return {
        "repo": os.path.basename(os.path.abspath(repo_path)),
        "stats": stats,
        "nodes": list(nodes.values()),
        "edges": edges,
    }


def filter_test_files(py_files: list[str]) -> list[str]:
    return [
        f
        for f in py_files
        if not (
            os.path.basename(f).startswith("test_")
            or os.path.basename(f).endswith("_test.py")
        )
    ]


def extract_one(repo_path: str, output_path: str, include_tests: bool) -> dict | None:
    """Extract a single repo's call graph and write it to output_path."""
    print(f"\n=== {os.path.basename(repo_path)} ===")
    print(f"Scanning: {repo_path}")
    py_files = find_python_files(repo_path)
    if not include_tests:
        py_files = filter_test_files(py_files)

    print(f"Found {len(py_files)} Python files")
    if not py_files:
        print("No Python files found, skipping.", file=sys.stderr)
        return None

    print("Running pyan3 static analysis...")
    try:
        graph = build_call_graph(py_files, repo_path)
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(graph, f, indent=2)

    s = graph["stats"]
    print(f"Done → {output_path}")
    print(
        f"  Nodes : {s['total_nodes']}  ({s['functions']} functions, {s['methods']} methods, {s['classes']} classes)"
    )
    print(f"  Edges : {s['total_edges']}")
    print(f"  Files : {s['files_analyzed']}")
    return graph


def main():
    parser = argparse.ArgumentParser(
        description="Extract a call graph from a Python repository using pyan3."
    )
    parser.add_argument(
        "repo_path",
        help=(
            "Path to a Python repo, a GitHub URL (cloned to a tempdir and "
            "removed afterwards), or — with --batch — a directory containing "
            "multiple repos."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="kg_output/call_graph.json",
        help="Output JSON file path for single-repo mode (default: kg_output/call_graph.json). Ignored in --batch mode.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Treat repo_path as a parent directory; process each subdirectory as a separate repo.",
    )
    parser.add_argument(
        "--output-dir",
        default="kg_output",
        help="Output directory for --batch mode (default: kg_output/). Each repo writes to <output-dir>/<repo>_call_graph.json.",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test files (default: excluded).",
    )
    args = parser.parse_args()

    if is_github_url(args.repo_path):
        if args.batch:
            print("Error: --batch is incompatible with a GitHub URL.", file=sys.stderr)
            sys.exit(1)
        repo = repo_name_from_url(args.repo_path)
        out_path = args.output
        if out_path == "kg_output/call_graph.json":
            out_path = os.path.join(args.output_dir, f"{repo}_call_graph.json")
        with cloned_repo(args.repo_path) as repo_dir:
            graph = extract_one(repo_dir, out_path, args.include_tests)
            if graph is None:
                sys.exit(1)
        return

    root_path = os.path.abspath(args.repo_path)
    if not os.path.isdir(root_path):
        print(f"Error: {root_path} is not a directory.", file=sys.stderr)
        sys.exit(1)

    if args.batch:
        repos = sorted(
            os.path.join(root_path, d)
            for d in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, d)) and not d.startswith(".")
        )
        if not repos:
            print(f"Error: no subdirectories found under {root_path}.", file=sys.stderr)
            sys.exit(1)

        print(f"Batch mode: {len(repos)} repos under {root_path}")
        results = []
        for repo in repos:
            out_path = os.path.join(
                args.output_dir, f"{os.path.basename(repo)}_call_graph.json"
            )
            graph = extract_one(repo, out_path, args.include_tests)
            results.append((os.path.basename(repo), graph))

        print("\n=== Batch summary ===")
        ok = sum(1 for _, g in results if g)
        print(f"Succeeded: {ok}/{len(results)}")
        for name, g in results:
            if g:
                s = g["stats"]
                print(
                    f"  {name:20s} {s['total_nodes']:4d} nodes  {s['total_edges']:4d} edges"
                )
            else:
                print(f"  {name:20s} FAILED")
    else:
        graph = extract_one(root_path, args.output, args.include_tests)
        if graph is None:
            sys.exit(1)


if __name__ == "__main__":
    main()
