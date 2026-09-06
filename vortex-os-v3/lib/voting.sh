# lib/voting.sh — Improvement #29: Confidence-Weighted Vote
# Multi-agent consensus: vote weight = agent.confidence
set -euo pipefail
LIB_PREFIX="[voting]"

voting_tally() {
  local answers_file="$1"  # NDJSON: each line {agent, answer, confidence}
  jq -s '
    group_by(.answer)
    | map({answer:.[0].answer, score:(map(.confidence)|add), count:length})
    | sort_by(-.score)
    | .[0]
  ' < "$answers_file"
}

voting_consensus_score() {
  local answers_file="$1"
  jq -s '
    [.[] | .confidence] as $c
    | ([$c[]] | add) as $sum
    | (max as $m | ($c | map(if . == $m then 1 else 0 end) | add)) as $winners
    | if $sum > 0 then $winners / length else 0 end
  ' < "$answers_file"
}
