"""File-backed store for saved eval cases.

Each `EvalCase` is persisted as one `<dir>/<sanitized id>.json` file so a
regression suite can be reloaded and re-run across restarts. The store is the
persistence layer behind the `/api/eval/cases` endpoints in `app.main`.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Union

from app.models.eval import EvalCase

logger = logging.getLogger(__name__)

# Runs of characters outside [a-z0-9-] collapse to a single dash.
_SANITIZE_RE = re.compile(r"[^a-z0-9-]+")


def _sanitize_id(case_id: str) -> str:
    """Reduce an id to a safe filename stem: lowercase [a-z0-9-]+.

    Non-conforming runs collapse to a single dash; leading/trailing dashes are
    stripped. Raises ValueError if nothing usable remains (blocks path
    traversal and empty filenames).
    """
    sanitized = _SANITIZE_RE.sub("-", case_id.lower()).strip("-")
    if not sanitized:
        raise ValueError(f"eval case id {case_id!r} sanitizes to empty")
    return sanitized


class EvalStore:
    """CRUD over eval-case JSON files in a single directory."""

    def __init__(self, directory: Union[str, Path]):
        self.directory = Path(directory)

    def list(self) -> List[EvalCase]:
        """Return all cases, sorted by id. Skips files that fail to parse."""
        if not self.directory.is_dir():
            return []

        cases: List[EvalCase] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    cases.append(EvalCase(**json.load(f)))
            except Exception as e:
                logger.warning("Skipping unparseable eval case %s: %s", path, e)
        cases.sort(key=lambda c: c.id)
        return cases

    def save(self, case: EvalCase) -> EvalCase:
        """Write (or replace) the case's JSON file. Returns the case."""
        stem = _sanitize_id(case.id)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{stem}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(case.model_dump(by_alias=True), f, indent=2)
        return case

    def delete(self, case_id: str) -> None:
        """Remove the case's JSON file. No error if it is already absent."""
        stem = _sanitize_id(case_id)
        (self.directory / f"{stem}.json").unlink(missing_ok=True)
