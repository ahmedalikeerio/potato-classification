#!/bin/bash
set -e
cd "$(dirname "$0")/../ui"
docker build -t potato-ui .
docker run --rm -p 8501:8080 -e API_URL=http://host.docker.internal:8080 potato-ui
