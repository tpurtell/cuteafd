#!/usr/bin/env bash
# Dev/test harness only. Provider choices are not shipped engine configuration.
# Usage: upstreams.sh deepseek|deepseek-anthropic|openrouter-mimo|litellm MODEL [gateway flags]
# Set CUTEAFD_BIN to a CPU build; credentials and private endpoints stay in env.
set -euo pipefail
. "$HOME/.cache/cuteafd/builds/api-gateway/secrets.sh"
provider=${1:?provider required}; shift
model=${1:-}; [[ $# -eq 0 ]] || shift
export CUTEAFD_TEST_PROVIDER=$provider
case "$provider" in
    deepseek)
        export CUTEAFD_TEST_URL=https://api.deepseek.com
        key_env=DEEPSEEK_API_KEY; model=${model:-deepseek-flash}
        flags=(--upstream-capabilities vision,reasoning --upstream-thinking-toggle)
        ;;
    deepseek-anthropic)
        export CUTEAFD_TEST_URL=https://api.deepseek.com/anthropic
        key_env=DEEPSEEK_API_KEY; model=${model:-deepseek-flash}
        flags=(--upstream-flavor anthropic --upstream-capabilities vision,reasoning --upstream-thinking-toggle)
        ;;
    openrouter-mimo)
        export CUTEAFD_TEST_URL=https://openrouter.ai/api/v1
        key_env=OPENROUTER_API_KEY; model=${model:-xiaomi/mimo-v2.6-pro}
        flags=(--upstream-capabilities vision,audio_in,reasoning)
        ;;
    litellm)
        export CUTEAFD_TEST_URL="${LITELLM_BASE_URL:?private base URL required}/v1"
        key_env=LITELLM_API_KEY
        case "$model" in
            claude/xiaomi/*|claude/qwen/*|claude/z-ai/*|claude/moonshotai/*|claude/deepseek/*) ;;
            *) printf 'model is not permitted for this test harness\n' >&2; exit 2 ;;
        esac
        flags=(--upstream-capabilities reasoning)
        ;;
    *) printf 'unknown test provider\n' >&2; exit 2 ;;
esac
exec "${CUTEAFD_BIN:-cuteafd}" gateway --upstream-url-env CUTEAFD_TEST_URL \
    --upstream-key-env "$key_env" --model "$model" "${flags[@]}" "$@"
