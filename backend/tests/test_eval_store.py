from app.models.eval import EvalCase
from app.models.route import RouteOptions
from app.services.eval_store import EvalStore


def _case(case_id="c1", name="n"):
    return EvalCase(
        id=case_id,
        name=name,
        start={"lat": 0, "lon": 0},
        end={"lat": 1, "lon": 1},
        options=RouteOptions(),
        referencePath=[{"lat": 0, "lon": 0}, {"lat": 1, "lon": 1}],
    )


def test_save_list_delete_roundtrip(tmp_path):
    store = EvalStore(tmp_path)
    store.save(_case())

    cases = store.list()
    assert len(cases) == 1
    assert cases[0].id == "c1"

    store.delete("c1")
    assert store.list() == []


def test_delete_missing_is_noop(tmp_path):
    store = EvalStore(tmp_path)
    # Should not raise even though nothing was ever saved.
    store.delete("never-existed")


def test_save_replaces_same_id(tmp_path):
    store = EvalStore(tmp_path)
    store.save(_case(name="first"))
    store.save(_case(name="second"))

    cases = store.list()
    assert len(cases) == 1
    assert cases[0].name == "second"


def test_list_sorted_by_id(tmp_path):
    store = EvalStore(tmp_path)
    store.save(_case(case_id="banana"))
    store.save(_case(case_id="apple"))

    assert [c.id for c in store.list()] == ["apple", "banana"]


def test_id_sanitization_roundtrips(tmp_path):
    store = EvalStore(tmp_path)
    store.save(_case(case_id="Half Dome!"))

    # File name is sanitized to [a-z0-9-]+.
    files = [p.name for p in tmp_path.glob("*.json")]
    assert files == ["half-dome.json"]

    # The stored id itself is preserved; only the filename is sanitized.
    cases = store.list()
    assert len(cases) == 1
    assert cases[0].id == "Half Dome!"


def test_list_skips_unparseable_files(tmp_path):
    store = EvalStore(tmp_path)
    store.save(_case())
    (tmp_path / "broken.json").write_text("{ not valid json", encoding="utf-8")

    cases = store.list()
    assert len(cases) == 1
    assert cases[0].id == "c1"


def test_list_missing_directory_returns_empty(tmp_path):
    store = EvalStore(tmp_path / "does-not-exist")
    assert store.list() == []
