#!/usr/bin/env sh

set -u

APP_MODULE="mlstudio/main.py"
SYNC_LOG="$(mktemp)"

is_network_error() {
    grep -Eiq "network|internet|connection|connect|timed out|timeout|dns|temporary failure|name resolution|failed to resolve|could not resolve|proxy|ssl|tls|certificate|offline" "$1"
}

echo "Running uv sync..."
if uv sync >"$SYNC_LOG" 2>&1; then
    cat "$SYNC_LOG"
else
    sync_status=$?
    if is_network_error "$SYNC_LOG"; then
        echo "uv sync failed because the network appears unavailable. Continuing with the existing environment."
        cat "$SYNC_LOG"
    else
        echo "uv sync failed with a non-network error:" >&2
        cat "$SYNC_LOG" >&2
        rm -f "$SYNC_LOG"
        exit "$sync_status"
    fi
fi

rm -f "$SYNC_LOG"

echo "Starting Streamlit..."
uv run -m streamlit run "$APP_MODULE"
run_status=$?

if [ "$run_status" -ne 0 ]; then
    echo "Streamlit failed to run. Check the error output above." >&2
    exit "$run_status"
fi
