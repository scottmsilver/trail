## Development Best Practices

- Do test driven development
- Lets check in to git with small improvements each step of the way
- Always use a virtual env for node
- **Python venv is `trail_env`** (repo root: `trail_env/`). Always run Python through it — never the system/anaconda `python`/`pytest`. Backend code and tests run from `backend/` as `PYTHONPATH=. ../trail_env/bin/python -m pytest ...`. Keep the backend deps installed in it (e.g. `geopy`).
- Pre-commit runs `./run_tests.sh` — a fast smoke suite (engine_v2 + path-module unit tests) under `trail_env`. The full/slow legacy suite belongs in CI, not the per-commit gate.
- Fix tests by fixing the logic or showing the test is invalid. Turning off a test to make it pass is against the rules.
- Tests should go into a tests directory. /tests
- Please run appropriate tests after each code change
- Put any debugging code into a debug directory. If it's temporary, use a tmp_debug directory

## Caching Strategy

- We don't do any caching for now. Fail requests when the data re not available. Require the caller to ask to load the data.
