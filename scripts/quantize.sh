
#!/usr/bin/env bash
#
# quantize.sh — APEX quantization for llama.cpp
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo ">>> $*"; }

# -----------------------------------
# DEFAULTS
# -----------------------------------
PROFILE="balanced"
CONFIG_FILE=""
IMATRIX=""
BASE_TYPE="Q6_K"
NUM_LAYERS="${NUM_LAYERS:-40}"
GENERATE_ONLY=false
CONFIG_OUTPUT=""
POSITIONAL=()

# -----------------------------------
# NEW: FULL PIPELINE SUPPORT
# -----------------------------------
FULL_PIPELINE=false
HF_MODEL=""

# -----------------------------------
# FIND LLAMA-QUANTIZE
# -----------------------------------
find_quantize() {
    if [ -n "${LLAMA_QUANTIZE:-}" ] && [ -f "$LLAMA_QUANTIZE" ]; then
        echo "$LLAMA_QUANTIZE"
        return 0
    fi

    local dirs=(
        "${LLAMA_CPP_DIR:-}"
        "./llama.cpp/build/bin"
        "$SCRIPT_DIR/../llama.cpp/build/bin"
    )

    for d in "${dirs[@]}"; do
        [ -n "$d" ] && [ -f "$d/llama-quantize" ] && echo "$d/llama-quantize" && return 0
    done

    command -v llama-quantize 2>/dev/null && return 0

    return 1
}

# -----------------------------------
# PARSE ARGS
# -----------------------------------
while [ $# -gt 0 ]; do
    case "$1" in

        --profile|-p)
            PROFILE="$2"
            shift 2
            ;;

        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;

        --imatrix|-i)
            IMATRIX="$2"
            shift 2
            ;;

        --base-type|-b)
            BASE_TYPE="$2"
            shift 2
            ;;

        --layers|-l)
            NUM_LAYERS="$2"
            shift 2
            ;;

        --generate-config)
            GENERATE_ONLY=true
            shift
            ;;

        -o)
            CONFIG_OUTPUT="$2"
            shift 2
            ;;

        # -----------------------------------
        # NEW: FULL PIPELINE MODE
        # -----------------------------------
        --full-pipeline)
            FULL_PIPELINE=true
            HF_MODEL="$2"
            shift 2
            ;;

        --help|-h)
            echo ""
            echo "Usage:"
            echo ""
            echo "  ./scripts/quantize.sh --profile balanced input.gguf output.gguf"
            echo ""
            echo "  OR"
            echo ""
            echo "  ./scripts/quantize.sh --full-pipeline Qwen/Qwen3.5-35B-A3B"
            echo ""
            exit 0
            ;;

        -*)
            die "Unknown option: $1"
            ;;

        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

# -----------------------------------
# PROFILE → BASE TYPE
# -----------------------------------
case "$PROFILE" in
    quality|i-quality)
        BASE_TYPE="Q6_K"
        ;;

    balanced|i-balanced)
        BASE_TYPE="Q6_K"
        ;;

    compact|i-compact)
        BASE_TYPE="Q4_K_M"
        ;;

    mini)
        BASE_TYPE="Q3_K_M"
        ;;
esac

# -----------------------------------
# FULL PIPELINE MODE
# HF -> F16 -> QUANT
# -----------------------------------
if $FULL_PIPELINE; then

    info "======================================="
    info "APEX FULL PIPELINE"
    info "======================================="

    info "HF MODEL: $HF_MODEL"

    LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/app/llama.cpp}"

    CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"

    [ -f "$CONVERT_SCRIPT" ] || die "convert_hf_to_gguf.py not found"

    MODEL_NAME=$(basename "$HF_MODEL" | tr '/' '-')

    F16_OUT="${MODEL_NAME}-f16.gguf"

    OUTPUT="${MODEL_NAME}-apex-${PROFILE}.gguf"

    info ""
    info "STEP 1: HF -> F16 GGUF"
    info "OUTPUT: $F16_OUT"
    info ""

    python3 "$CONVERT_SCRIPT" \
        "$HF_MODEL" \
        --outtype f16 \
        --outfile "$F16_OUT"

    [ -f "$F16_OUT" ] || die "F16 conversion failed"

    info ""
    info "F16 GENERATED: $F16_OUT"
    info ""

    POSITIONAL=("$F16_OUT" "$OUTPUT")
fi

# -----------------------------------
# I-PROFILES WARNING
# -----------------------------------
case "$PROFILE" in
    i-quality|i-balanced|i-compact|mini)
        if [ -z "$IMATRIX" ]; then
            echo "WARNING: Profile '$PROFILE' benefits from --imatrix"
        fi
        ;;
esac

# -----------------------------------
# CONFIG GENERATION
# -----------------------------------
if [ -n "$CONFIG_FILE" ]; then

    [ -f "$CONFIG_FILE" ] || die "Config file not found: $CONFIG_FILE"

    TTFILE="$CONFIG_FILE"

    info "Using config: $CONFIG_FILE"

elif $GENERATE_ONLY; then

    if [ -n "$CONFIG_OUTPUT" ]; then
        "$SCRIPT_DIR/generate_config.sh" \
            --profile "$PROFILE" \
            --layers "$NUM_LAYERS" \
            -o "$CONFIG_OUTPUT"
    else
        "$SCRIPT_DIR/generate_config.sh" \
            --profile "$PROFILE" \
            --layers "$NUM_LAYERS"
    fi

    exit 0

else

    TTFILE="$(mktemp)"

    trap 'rm -f "$TTFILE"' EXIT

    "$SCRIPT_DIR/generate_config.sh" \
        --profile "$PROFILE" \
        --layers "$NUM_LAYERS" \
        -o "$TTFILE"

    info "Generated config for profile '$PROFILE'"
fi

# -----------------------------------
# GENERATE ONLY EXIT
# -----------------------------------
$GENERATE_ONLY && exit 0

# -----------------------------------
# INPUT / OUTPUT
# -----------------------------------
[ ${#POSITIONAL[@]} -ge 2 ] || die "Usage: $0 --profile <profile> <input.gguf> <output.gguf>"

INPUT="${POSITIONAL[0]}"
OUTPUT="${POSITIONAL[1]}"

[ -f "$INPUT" ] || die "Input file not found: $INPUT"

# -----------------------------------
# FIND QUANTIZE
# -----------------------------------
QUANTIZE=$(find_quantize) || die "llama-quantize not found"

# -----------------------------------
# BUILD COMMAND
# -----------------------------------
QUANT_ARGS=(
    "--tensor-type-file" "$TTFILE"
)

[ -n "$IMATRIX" ] && QUANT_ARGS+=(
    "--imatrix" "$IMATRIX"
)

# -----------------------------------
# LOGS
# -----------------------------------
info ""
info "======================================="
info "APEX QUANTIZATION"
info "======================================="
info "Profile:    $PROFILE"
info "Base Type:  $BASE_TYPE"
info "Input:      $INPUT"
info "Output:     $OUTPUT"

[ -n "$IMATRIX" ] && info "Imatrix:    $IMATRIX"

info "Config:     $TTFILE"
info ""

# -----------------------------------
# RUN QUANTIZATION
# -----------------------------------
"$QUANTIZE" \
    "${QUANT_ARGS[@]}" \
    "$INPUT" \
    "$OUTPUT" \
    "$BASE_TYPE"

# -----------------------------------
# DONE
# -----------------------------------
info ""
info "======================================="
info "DONE"
info "======================================="
info "Output: $OUTPUT"

ls -lh "$OUTPUT"

