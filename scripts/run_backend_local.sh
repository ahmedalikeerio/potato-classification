#!/bin/bash
set -e
cd "$(dirname "$0")/../backend"
docker build -t potato-backend .
docker run --rm -p 8080:8080 potato-backend
