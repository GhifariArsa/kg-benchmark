"""
Stage 3a: Agent runner (Track A — product comparison).

Runs each question in a question_bank.json against one or more CLI coding
agents shelled out as subprocesses. Each agent uses its native tool surface
(Read/Grep/Bash/etc.) — no KG MCP, no graph access. The agent works in
`--repo-dir` as cwd.

Output: agent_runs.json with raw transcripts + timing. Grading happens
separately in scorer.py.
"""

import argparse
import json
import os
import subprocess
import time


def _parse_claude_code_json(stdout):
    try:
        data = json.loads(stdout)
        return data.get("result") or data.get("text") or stdout
    except Exception:
        return stdout


AGENTS = {
    "claude-code": {
        "cmd": ["claude", "-p", "--output-format", "json"],
        "parse": _parse_claude_code_json,
    },
    "codex": {
        "cmd": ["codex", "exec"],
        "parse": str.strip,
    },
    "opencode": {
        "cmd": [
            "opencode", "run",
            "--model", "nectar/qwen3.6:35b-a3b",
            "--agent", "question-and-answer-no-mcp",
        ],
        "parse": str.strip,
    },
}

DEFAULT_TIMEOUT_S = 300

PROMPT_TEMPLATE = """Answer the following code-comprehension question about the codebase in the current working directory.

Use the tools available to you (file reading, grep, etc.) to investigate the source. Your final answer MUST be a single fully-qualified Python identifier (e.g. `module.Class.method` or `module.function`) that exactly matches a name as it appears in the source.

End your response with a line of the form:
Answer: <qualified_name>

Question: {question}
"""


def run_one(agent_name, question, repo_dir, timeout_s):
    spec = AGENTS[agent_name]
    prompt = PROMPT_TEMPLATE.format(question=question["question"])

    base = {
        "question_id": question["id"],
        "agent": agent_name,
        "repo_dir": repo_dir,
    }

    start = time.time()
    try:
        proc = subprocess.run(
            spec["cmd"] + [prompt],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {**base, "completed": False, "agent_answer": None,
                "elapsed_s": timeout_s, "error": "timeout"}
    except FileNotFoundError as e:
        return {**base, "completed": False, "agent_answer": None,
                "error": f"CLI not found: {e}"}

    ok = proc.returncode == 0
    return {
        **base,
        "completed": ok,
        "agent_answer": spec["parse"](proc.stdout) if ok else None,
        "raw_stdout": (proc.stdout or "")[-4000:],
        "raw_stderr": (proc.stderr or "")[-2000:],
        "exit_code": proc.returncode,
        "elapsed_s": round(time.time() - start, 2),
    }


def load_cache(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return {(r["question_id"], r["agent"]): r for r in data.get("runs", [])}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Run a question bank through CLI coding agents.")
    parser.add_argument("question_bank")
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--agents", nargs="+", default=["claude-code"], choices=list(AGENTS))
    parser.add_argument("-o", "--output", default="qa_output/agent_runs.json")
    parser.add_argument("-n", "--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.repo_dir):
        raise SystemExit(f"--repo-dir is not a directory: {args.repo_dir}")

    with open(args.question_bank) as f:
        bank = json.load(f)
    questions = bank["questions"][: args.limit] if args.limit else bank["questions"]

    cache = {} if args.no_cache else load_cache(args.output)
    tasks = [(a, q) for q in questions for a in args.agents if (q["id"], a) not in cache]

    print(f"{len(tasks)} runs queued ({len(cache)} cached). Agents: {args.agents}. Timeout: {args.timeout}s.")

    runs = list(cache.values())
    for a, q in tasks:
        r = run_one(a, q, args.repo_dir, args.timeout)
        runs.append(r)
        mark = "✓" if r.get("completed") else "✗"
        ans = (r.get("agent_answer") or r.get("error") or "")[:80].replace("\n", " ")
        print(f"  {mark} [{r['agent']:12s}] {r['question_id']}  ({r.get('elapsed_s', '?')}s)  {ans}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "source_bank": args.question_bank,
            "repo_dir": args.repo_dir,
            "agents": args.agents,
            "num_runs": len(runs),
            "runs": runs,
        }, f, indent=2)

    n_ok = sum(1 for r in runs if r.get("completed"))
    print(f"\nSaved {len(runs)} runs → {args.output}  ({n_ok} completed, {len(runs) - n_ok} failed)")


if __name__ == "__main__":
    main()
