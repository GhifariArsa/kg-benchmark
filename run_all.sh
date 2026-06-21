#!/usr/bin/env bash
# End-to-end pipeline: graph extraction → question generation → agent answering → scoring.
#
# Stage 1 (kg_extractor_v2.py)   : pyan-based call graph per repo  → output/
# Stage 2 (batch_qa.py)          : LLM-generated question banks    → qa_dataset/   (OpenAI gpt-4o-mini)
# Stage 3 (agent_runner.py)      : opencode + local Ollama answers → qa_output/
# Stage 4 (scorer.py)            : LLM-as-judge grading            → qa_output/    (OpenAI gpt-4o-mini)
#
# Requires: uv, opencode CLI, a local ollama server with `qwen3.6:35b-a3b` pulled,
# and OPENAI_API_KEY set for stages 2 and 4.

set -euo pipefail

REPOS_DIR="${REPOS_DIR:-test_repositories}"
GRAPH_DIR="${GRAPH_DIR:-output}"
BANK_DIR="${BANK_DIR:-qa_dataset}"
RUNS_DIR="${RUNS_DIR:-qa_output}"
AGENT="${AGENT:-opencode}"
TIMEOUT="${TIMEOUT:-300}"

mkdir -p "$GRAPH_DIR" "$BANK_DIR" "$RUNS_DIR"

echo "=== Stage 1: extract call graphs ==="
uv run python kg_extractor_v2.py "$REPOS_DIR" --batch --output-dir "$GRAPH_DIR"

echo
echo "=== Stage 2: generate question banks ==="
uv run python batch_qa.py --graph-dir "$GRAPH_DIR" --out-dir "$BANK_DIR" --skip-existing

echo
echo "=== Stages 3 & 4: run agents and score per repo ==="
for repo_path in "$REPOS_DIR"/*/; do
  repo="$(basename "$repo_path")"
  bank="$BANK_DIR/${repo}_question_bank.json"
  runs="$RUNS_DIR/${repo}_agent_runs.json"
  scored="$RUNS_DIR/${repo}_scored_results.json"

  if [[ ! -f "$bank" ]]; then
    echo "⊘ $repo: no question bank ($bank), skipping"
    continue
  fi

  echo
  echo "▶ $repo"
  uv run python agent_runner.py "$bank" \
    --repo-dir "$repo_path" \
    --agents "$AGENT" \
    --timeout "$TIMEOUT" \
    -o "$runs"

  uv run python scorer.py "$runs" "$bank" -o "$scored"
done

echo
echo "=== Done. Scored results in $RUNS_DIR/*_scored_results.json ==="
