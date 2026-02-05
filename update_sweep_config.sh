#!/usr/bin/env bash
set -euo pipefail

# Apply each top-level parameter from params.yaml into sweep.yaml one-by-one.
# For each param:
#   - restore original sweep.yaml
#   - inject only that param under .parameters
#   - run python3 main.py
#
# Usage:
#   ./apply_params_one_by_one.sh sweep.yaml params.yaml
#   ./apply_params_one_by_one.sh sweep.yaml params.yaml -- --any --args for main.py

if ! command -v yq >/dev/null 2>&1; then
  echo "Error: yq not found (need mikefarah/yq v4+)." >&2
  exit 1
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <sweep.yaml> <params.yaml> [-- python_args]" >&2
  exit 1
fi

SWEEP_YAML="$1"
PARAMS_YAML="$2"
shift 2

PY_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  PY_ARGS=("$@")
fi

if [[ ! -f "$SWEEP_YAML" ]]; then
  echo "Error: sweep file not found: $SWEEP_YAML" >&2
  exit 1
fi
if [[ ! -f "$PARAMS_YAML" ]]; then
  echo "Error: params file not found: $PARAMS_YAML" >&2
  exit 1
fi

# Backup the original sweep
SWEEP_BAK="$(mktemp)"
cp "$SWEEP_YAML" "$SWEEP_BAK"
trap 'cp "$SWEEP_BAK" "$SWEEP_YAML"; rm -f "$SWEEP_BAK"' EXIT

# List top-level keys in params.yaml (learning_rate, batch_size, ...)
mapfile -t PARAM_KEYS < <(yq -r 'keys | .[]' "$PARAMS_YAML")

if [[ ${#PARAM_KEYS[@]} -eq 0 ]]; then
  echo "Error: no top-level keys found in $PARAMS_YAML" >&2
  exit 1
fi

for key in "${PARAM_KEYS[@]}"; do
  echo
  echo "=============================="
  echo "▶ Applying param: $key"
  echo "=============================="

  # Restore original sweep before each run (non-cumulative)
  cp "$SWEEP_BAK" "$SWEEP_YAML"

  # Create a tiny YAML mapping with ONLY this key
  TMP_ONE="$(mktemp)"
  yq e ". | {\"$key\": .[\"$key\"]}" "$PARAMS_YAML" > "$TMP_ONE"

  # Merge it into .parameters
  yq -i '
    .parameters = (.parameters // {}) |
    .parameters = (.parameters * load("'"$TMP_ONE"'"))
  ' "$SWEEP_YAML"

  rm -f "$TMP_ONE"

  echo "✅ Updated sweep.yaml with only '$key'. Running:"
  echo "   python3 main.py ${PY_ARGS[*]:-}"

  # Optional: pass which param is being tested (handy inside main.py)
  CURRENT_PARAM="$key" python3 main.py --sweep True "${PY_ARGS[@]}"
done

echo
echo "🏁 All params applied one-by-one. sweep.yaml restored to original."
