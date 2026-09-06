#!/usr/bin/env bash
# Example: Competitive Landscape Report
# Usage:
#   cd examples/01-competitive-landscape
#   bash run.sh
set -euo pipefail
SKILL="${SKILL_PATH:-../../skill.sh}"
CONFIG="./config.json"
ARGS=(
  --config "$CONFIG"
)

# 1. Set goal
"$SKILL" "${ARGS[@]}" --plan
GOAL="Research the top 3 competitors in the AI agent orchestration space and synthesize a landscape report."
python3 -c "
import json, sys
p = json.load(open('plan.json'))
p['goal'] = sys.argv[1]
open('plan.json','w').write(json.dumps(p, indent=2))
" "$GOAL"

# 2. Apply the competitor_landscape template with vars
"$SKILL" "${ARGS[@]}" --import-template competitor_landscape \
    --template-var INDUSTRY="AI agent orchestration" \
    --template-var PRODUCT="Competitor Landscape"

# 3. Validate
"$SKILL" "${ARGS[@]}" --validate

# 4. Inspect ready tasks
"$SKILL" "${ARGS[@]}" --ready

# 5. Dispatch (the parent agent would execute them in parallel here)
"$SKILL" "${ARGS[@]}" --dispatch-ready

# 6. Simulate three research outputs
mkdir -p ./out
for i in 1 2 3; do
  echo "# Competitor $i placeholder" > "./out/comp_$i.md"
  "$SKILL" "${ARGS[@]}" --mark-done "r_competitor_$i" "./out/comp_$i.md"
done

# 7. Simulate the synthesis writer
"$SKILL" "${ARGS[@]}" --dispatch-ready
echo "# Landscape report placeholder" > "./out/landscape.md"
"$SKILL" "${ARGS[@]}" --mark-done "w_landscape_report" "./out/landscape.md"

# 8. Inspect stats and seal
"$SKILL" "${ARGS[@]}" --stats
"$SKILL" "${ARGS[@]}" --aggregate
"$SKILL" "${ARGS[@]}" --seal
