#!/usr/bin/env bash
set -euo pipefail

DHAT_FILE="${1:-dhat-heap.json}"

if [[ ! -f "$DHAT_FILE" ]]; then
    echo "error: dhat file not found: $DHAT_FILE"
    exit 1
fi

VIEWER_URL="https://nnethercote.github.io/dh_view/dh_view.html"

echo "opening dhat viewer in browser..."
echo "viewer: $VIEWER_URL"
echo "profile: $DHAT_FILE"
echo ""
echo "instructions:"
echo "1. browser will open to dhat viewer"
echo "2. click 'load...' button"
echo "3. select: $DHAT_FILE"
echo ""

# open viewer in browser
if command -v xdg-open &> /dev/null; then
    xdg-open "$VIEWER_URL"
elif command -v open &> /dev/null; then
    open "$VIEWER_URL"
else
    echo "please open this url manually: $VIEWER_URL"
fi
