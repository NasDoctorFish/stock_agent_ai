#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: ./manage.sh [dev|build|start|docker]"
}

case "${1:-}" in
  dev)
    npm run dev
    ;;
  build)
    npm run build
    ;;
  start)
    npm run start
    ;;
  docker)
    docker compose up --build
    ;;
  *)
    usage
    exit 1
    ;;
esac
