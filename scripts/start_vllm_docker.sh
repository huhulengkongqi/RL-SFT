#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Building vLLM Docker image..."
docker compose build vllm

echo "Starting vLLM server in Docker..."
docker compose up -d vllm

echo "Waiting for server to be ready..."
sleep 10

echo "Server logs:"
docker compose logs -f vllm
