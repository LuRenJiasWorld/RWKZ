#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# generate_quants.sh — Generate all RWKZ quantization levels from F16 source
# ---------------------------------------------------------------------------
# Prerequisites:
#   - llama.cpp tools (llama-imatrix + llama-quantize) in PATH or set
#     LLAMA_CPP_BIN_DIR below
#   - F16 source model at models/rwkv7-0.1b-g1-f16.gguf
#   - Calibration text for imatrix (auto-downloaded or use local file)
#
# Usage:
#   ./scripts/quantization/generate_quants.sh              # generate all levels
#   ./scripts/quantization/generate_quants.sh IQ2_XXS     # generate single level
#   LLAMA_CPP_BIN_DIR=/tmp/llama.cpp/build/bin ./scripts/quantization/generate_quants.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"

# ─── Configuration ──────────────────────────────────────────────────────────

LLAMA_CPP_BIN_DIR="${LLAMA_CPP_BIN_DIR:-}"
MODEL_NAME="rwkv7-0.1b-g1"
F16_SOURCE="$MODELS_DIR/${MODEL_NAME}-f16.gguf"

# Calibration dataset: wiki.train.raw from ikawrakow/validation-datasets-for-llama.cpp
CALIBRATION_REPO="ikawrakow/validation-datasets-for-llama.cpp"
CALIBRATION_FILE="wiki.train.raw.gz"
CALIBRATION_LOCAL="$MODELS_DIR/.calibration"

# Imatrix cache: reuse across runs
IMATRIX_FILE="$MODELS_DIR/.imatrix/${MODEL_NAME}.imatrix"

# ─── Quantization levels to generate ────────────────────────────────────────
# Each entry: "QUANT_TYPE  output_suffix"
# Order: smallest to largest

# Legacy quant types (no imatrix needed)
LEGACY_QUANTS=(
    "Q4_0  q4_0"
    "Q4_1  q4_1"
    "Q5_0  q5_0"
    "Q5_1  q5_1"
    "Q8_0  q8_0"
    "F16   f16"
)

# K-quant types (no imatrix needed, but benefit from it)
K_QUANTS=(
    "Q2_K    q2_k"
    "Q3_K_S  q3_k_s"
    "Q3_K_M  q3_k_m"
    "Q3_K_L  q3_k_l"
    "Q4_K_S  q4_k_s"
    "Q4_K_M  q4_k_m"
    "Q5_K_S  q5_k_s"
    "Q5_K_M  q5_k_m"
    "Q6_K    q6_k"
)

# IQ quant types (REQUIRE importance matrix)
IQ_QUANTS=(
    "IQ2_XXS  iq2_xxs"
    "IQ2_XS   iq2_xs"
    "IQ2_S    iq2_s"
    "IQ3_XXS  iq3_xxs"
    "IQ3_XS   iq3_xs"
    "IQ3_S    iq3_s"
    "IQ4_XS   iq4_xs"
    "IQ4_NL   iq4_nl"
)

# ─── Helpers ────────────────────────────────────────────────────────────────

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }

find_tool() {
    local tool="$1"
    # Check explicit path first
    if [ -n "$LLAMA_CPP_BIN_DIR" ] && [ -x "$LLAMA_CPP_BIN_DIR/$tool" ]; then
        echo "$LLAMA_CPP_BIN_DIR/$tool"
        return
    fi
    # Check PATH
    if command -v "$tool" &>/dev/null; then
        command -v "$tool"
        return
    fi
    # Check common build locations
    for dir in /tmp/llama.cpp/build/bin ./llama.cpp/build/bin; do
        if [ -x "$dir/$tool" ]; then
            echo "$dir/$tool"
            return
        fi
    done
    die "$tool not found. Set LLAMA_CPP_BIN_DIR or install llama.cpp"
}

quantize_one() {
    local quant_type="$1"
    local file_suffix="$2"
    local output="$MODELS_DIR/${MODEL_NAME}-${file_suffix}.gguf"
    local imatrix_flag=""

    if [ -f "$output" ]; then
        info "Skipping $quant_type — already exists: $output"
        return
    fi

    # Use imatrix for IQ types, optionally for K types
    if [ -f "$IMATRIX_FILE" ]; then
        imatrix_flag="--imatrix $IMATRIX_FILE"
    fi

    info "Generating $quant_type → $output"
    "$LLAMA_QUANTIZE" $imatrix_flag "$F16_SOURCE" "$output" "$quant_type"
    info "  Size: $(du -h "$output" | cut -f1)"
}

# ─── Main ───────────────────────────────────────────────────────────────────

LLAMA_QUANTIZE=$(find_tool llama-quantize)
LLAMA_IMATRIX=$(find_tool llama-imatrix)

[ -f "$F16_SOURCE" ] || die "F16 source not found: $F16_SOURCE"

# Determine which levels to generate
TARGET="${1:-all}"

# ─── Step 1: Prepare calibration data & generate imatrix (for IQ quants) ────

if [ "$TARGET" = "all" ] || [[ "$TARGET" == IQ* ]]; then
    if [ ! -f "$IMATRIX_FILE" ]; then
        info "Preparing calibration dataset..."

        mkdir -p "$(dirname "$IMATRIX_FILE")"

        if [ ! -f "$CALIBRATION_LOCAL/wiki.train.raw" ]; then
            mkdir -p "$CALIBRATION_LOCAL"
            info "Downloading $CALIBRATION_FILE from $CALIBRATION_REPO..."
            hf download "$CALIBRATION_REPO" "$CALIBRATION_FILE" --repo-type dataset
            cp ~/.cache/huggingface/hub/datasets--ikawrakow--validation-datasets-for-llama.cpp/snapshots/*/wiki.train.raw.gz \
                "$CALIBRATION_LOCAL/wiki.train.raw.gz" 2>/dev/null || true
            gzip -d -c "$CALIBRATION_LOCAL/wiki.train.raw.gz" > "$CALIBRATION_LOCAL/wiki.train.raw"
        fi

        # Use first 500KB for imatrix (sufficient for 0.1B model)
        CALIBRATION_TXT="$CALIBRATION_LOCAL/calibration.txt"
        if [ ! -f "$CALIBRATION_TXT" ]; then
            head -c 500000 "$CALIBRATION_LOCAL/wiki.train.raw" > "$CALIBRATION_TXT"
            info "Calibration data: $(wc -c < "$CALIBRATION_TXT") bytes, $(wc -l < "$CALIBRATION_TXT") lines"
        fi

        info "Generating importance matrix (this takes ~30-40 minutes for 0.1B)..."
        "$LLAMA_IMATRIX" \
            -m "$F16_SOURCE" \
            -f "$CALIBRATION_TXT" \
            -o "$IMATRIX_FILE" \
            -c 512

        info "Imatrix saved to $IMATRIX_FILE"
    else
        info "Using cached imatrix: $IMATRIX_FILE"
    fi
fi

# ─── Step 2: Generate quantized models ──────────────────────────────────────

info "Starting quantization..."

generate_group() {
    local group_name="$1"
    shift
    local entries=("$@")

    for entry in "${entries[@]}"; do
        read -r quant_type file_suffix <<< "$entry"
        if [ "$TARGET" = "all" ] || [ "$quant_type" = "$TARGET" ]; then
            quantize_one "$quant_type" "$file_suffix"
        fi
    done
}

if [ "$TARGET" = "all" ]; then
    generate_group "Legacy" "${LEGACY_QUANTS[@]}"
    generate_group "K-quant" "${K_QUANTS[@]}"
    generate_group "IQ-quant" "${IQ_QUANTS[@]}"
else
    # Find the matching quant in one of the groups
    for group in LEGACY_QUANTS K_QUANTS IQ_QUANTS; do
        eval "local entries=(\"\${${group}[@]}\")"
        for entry in "${entries[@]}"; do
            read -r quant_type file_suffix <<< "$entry"
            if [ "$quant_type" = "$TARGET" ]; then
                quantize_one "$quant_type" "$file_suffix"
                break 2
            fi
        done
    done
fi

info "Done! Models in $MODELS_DIR"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null | awk '{print $5, $NF}'
