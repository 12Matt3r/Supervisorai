# lib/schema.sh — JSON Schema validation with a pure-Python fallback
# shellcheck shell=bash

[[ -n "${__LIB_SCHEMA_LOADED:-}" ]] && return 0
__LIB_SCHEMA_LOADED=1

# We deliberately avoid requiring jsonschema / jq. We ship a tiny
# pure-Python validator that handles the subset we use: required,
# properties, type, enum, minimum, maximum, items, additionalProperties.
# For anything else, an external `jsonschema` install is preferred.

# Returns 0 on valid, 1 on invalid, 2 on error (e.g. unreadable).
validate_json_file() {
  local file="$1" schema="$2"
  [[ -f "$file" ]] || { err "validate: missing $file"; return 2; }
  [[ -f "$schema" ]] || { err "validate: missing $schema"; return 2; }

  if python3 -c "import jsonschema" >/dev/null 2>&1; then
    python3 - "$file" "$schema" <<'PY'
import json, sys
from jsonschema import Draft7Validator, FormatChecker
data = json.load(open(sys.argv[1]))
schema = json.load(open(sys.argv[2]))
v = Draft7Validator(schema, format_checker=FormatChecker())
errors = list(v.iter_errors(data))
if errors:
    print("VALIDATION FAILED:")
    for e in errors:
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        print(f"  - {path}: {e.message}")
    sys.exit(1)
sys.exit(0)
PY
    return $?
  fi

  # Fallback: our pure-Python validator
  python3 - "$file" "$schema" <<'PY'
import json, re, sys
try:
    data = json.load(open(sys.argv[1]))
    schema = json.load(open(sys.argv[2]))
except Exception as e:
    print(f"Cannot read files: {e}", file=sys.stderr); sys.exit(2)

errors = []

def fails(cond, path, msg):
    if cond: errors.append(f"{path}: {msg}")

def check(value, sch, path="$"):
    if isinstance(sch, bool):
        if not sch and value:
            errors.append(f"{path}: schema forbids this value")
        return
    if not isinstance(sch, dict):
        return
    typ = sch.get("type")
    if typ:
        py_type = {
            "string": str, "integer": int, "number": (int, float),
            "boolean": bool, "array": list, "object": dict,
            "null": type(None),
        }
        expected = py_type.get(typ)
        if expected is None:
            errors.append(f"{path}: unknown type '{typ}'")
        elif typ == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"{path}: expected integer, got {type(value).__name__}")
        elif not isinstance(value, expected):
            errors.append(f"{path}: expected {typ}, got {type(value).__name__}")

    if "enum" in sch and value not in sch["enum"]:
        errors.append(f"{path}: value not in enum {sch['enum']}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in sch and value < sch["minimum"]:
            errors.append(f"{path}: < minimum {sch['minimum']}")
        if "maximum" in sch and value > sch["maximum"]:
            errors.append(f"{path}: > maximum {sch['maximum']}")

    if isinstance(value, str):
        if "pattern" in sch:
            if not re.search(sch["pattern"], value):
                errors.append(f"{path}: doesn't match pattern {sch['pattern']}")
        if "minLength" in sch and len(value) < sch["minLength"]:
            errors.append(f"{path}: shorter than minLength")

    if isinstance(value, list):
        if "items" in sch:
            for i, item in enumerate(value):
                check(item, sch["items"], f"{path}[{i}]")

    if isinstance(value, dict):
        for req in sch.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required field '{req}'")
        for k, v in sch.get("properties", {}).items():
            if k in value:
                check(value[k], v, f"{path}.{k}" if path == "$" else f"{path}.{k}")
        # additionalProperties
        ap = sch.get("additionalProperties", True)
        if ap is False:
            allowed = set(sch.get("properties", {}).keys())
            for k in value:
                if k not in allowed:
                    errors.append(f"{path}: unexpected property '{k}'")

check(data, schema)
if errors:
    print("VALIDATION FAILED:")
    for e in errors: print(f"  - {e}")
    sys.exit(1)
sys.exit(0)
PY
}

# Validate any JSON string against a schema without writing to disk.
validate_json_string() {
  local json="$1" schema="$2"
  local tmp; tmp=$(mktemp --suffix=.json)
  echo "$json" > "$tmp"
  validate_json_file "$tmp" "$schema"
  local rc=$?
  rm -f "$tmp"
  return $rc
}
