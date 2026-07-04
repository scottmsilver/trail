#!/usr/bin/env bash
# Fast pre-commit smoke tests for the trail backend.
#
# Runs under the project venv (trail_env) — NOT the anaconda base interpreter,
# and NOT the stray root pytest.ini (which belongs to a different project).
# Scope is deliberately the fast, self-contained engine_v2 + path-module unit
# tests (~1s). The full suite — including slow/legacy integration tests that
# hit the network — is a CI concern, not something to run on every commit.
set -euo pipefail

cd "$(dirname "$0")/backend" || exit 1

PYTHONPATH=. ../trail_env/bin/python -m pytest \
  tests/unit/test_path_expansion.py \
  tests/unit/test_path_connectivity.py \
  tests/unit/test_service_v2.py \
  tests/unit/test_path_layer.py \
  tests/unit/test_pathfinder_v2.py \
  tests/unit/test_engine_v2_imports.py \
  -q
