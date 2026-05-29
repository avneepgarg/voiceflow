#!/bin/bash
# Push VoiceFlow to GitHub
# Usage: GITHUB_TOKEN=*** ./scripts/push_to_github.sh

set -e

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: Set GITHUB_TOKEN environment variable"
    echo "Usage: GITHUB_TOKEN=*** ./scripts/push_to_github.sh"
    exit 1
fi

cd /home/avneep/voiceflow

echo "=== Setting git remote ==="
git remote set-url origin "https://avneepgarg:${GITHUB_TOKEN}@github.com/avneepgarg/voiceflow.git"

echo "=== Pushing to GitHub ==="
git push -u origin main

echo "=== Done ==="
echo "Check CI at: https://github.com/avneepgarg/voiceflow/actions"
