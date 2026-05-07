#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$ROOT_DIR/Project"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON:-python3}"
KERNEL_DIR="$PROJECT_DIR/kernels"
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"

NAIF_TLS_URL="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls"
DE442S_URL="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de442s.bsp"

download_file() {
    local url="$1"
    local output_path="$2"
    local tmp_path="${output_path}.tmp"

    if [[ -s "$output_path" ]]; then
        echo "Found existing ${output_path#"$ROOT_DIR"/}"
        return
    fi

    echo "Downloading $(basename "$output_path")"
    curl -L --fail --show-error --output "$tmp_path" "$url"
    mv "$tmp_path" "$output_path"
}

if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for kernel downloads." >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "$PYTHON_BIN was not found. Set PYTHON=/path/to/python3 if needed." >&2
    exit 1
fi

mkdir -p "$KERNEL_DIR" "$PROJECT_DIR/results"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating virtual environment at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Using existing virtual environment at $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

if [[ "${SKIP_CYIPOPT:-0}" == "1" ]]; then
    tmp_requirements="$(mktemp)"
    grep -vE '^[[:space:]]*cyipopt[[:space:]]*$' "$REQUIREMENTS_FILE" > "$tmp_requirements"
    "$VENV_PYTHON" -m pip install -r "$tmp_requirements"
    rm -f "$tmp_requirements"
else
    if ! "$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"; then
        cat >&2 <<'EOF'

Dependency install failed. If cyipopt is the blocker and you do not need it for
the simulation run, retry with:

    SKIP_CYIPOPT=1 ./setup_simulation.sh

EOF
        exit 1
    fi
fi

download_file "$NAIF_TLS_URL" "$KERNEL_DIR/naif0012.tls"
download_file "$DE442S_URL" "$KERNEL_DIR/de442s.bsp"

"$VENV_PYTHON" - <<PY
from pathlib import Path
import spiceypy as spice

root = Path("$ROOT_DIR")
kernels = [
    root / "Project" / "kernels" / "naif0012.tls",
    root / "Project" / "kernels" / "de442s.bsp",
]
for kernel in kernels:
    spice.furnsh(str(kernel))

state_km, _ = spice.spkpos("SUN", 0.0, "J2000", "NONE", "EARTH")
distance_km = sum(component * component for component in state_km) ** 0.5
print(f"SPICE smoke check passed: Sun-Earth distance at J2000 is {distance_km:.3f} km")
spice.kclear()
PY

cat <<EOF

Setup complete.

Run the simulation with:

    cd "$ROOT_DIR"
    source "$VENV_DIR/bin/activate"
    python Project/main.py

Or run it immediately with:

    ./setup_simulation.sh --run

EOF

if [[ "${1:-}" == "--run" ]]; then
    export MPLBACKEND="${MPLBACKEND:-Agg}"
    cd "$ROOT_DIR"
    "$VENV_PYTHON" "$PROJECT_DIR/main.py"
fi
