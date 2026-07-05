import app.main as main
from app.services.eval_store import EvalStore
from fastapi.testclient import TestClient


def _client(tmp_path, monkeypatch):
    # Point the module-level store at an isolated tmp dir; no `with` block so
    # startup events (DEM preloading etc.) do not run.
    monkeypatch.setattr(main, "eval_store", EvalStore(tmp_path))
    return TestClient(main.app)


def _case_payload(case_id="c1", name="My Case"):
    return {
        "id": case_id,
        "name": name,
        "start": {"lat": 0, "lon": 0},
        "end": {"lat": 1, "lon": 1},
        "options": {},
        "referencePath": [{"lat": 0, "lon": 0}, {"lat": 1, "lon": 1}],
    }


def test_post_get_delete_cases(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    # POST creates the case and echoes the id.
    resp = client.post("/api/eval/cases", json=_case_payload())
    assert resp.status_code == 200
    assert resp.json()["id"] == "c1"

    # GET lists it.
    resp = client.get("/api/eval/cases")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["id"] == "c1"

    # DELETE removes it.
    resp = client.delete("/api/eval/cases/c1")
    assert resp.status_code == 204

    # GET is now empty.
    resp = client.get("/api/eval/cases")
    assert resp.status_code == 200
    assert resp.json() == []


def test_post_invalid_id_returns_400(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.post("/api/eval/cases", json=_case_payload(case_id="!!!"))
    assert resp.status_code == 400
