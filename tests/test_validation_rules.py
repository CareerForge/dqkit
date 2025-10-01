
import pandas as pd
from dqkit.types import Dataset
from dqkit.validation.validation import validate, build_spec

def test_min_max_unique_and_dtype(tmp_path):
    df = pd.DataFrame({
        "age": [10, 20, 30, None],
        "price": [1.5, 2.0, 3.0, 4.0],
        "code": ["A1", "A2", "B3", "C4"],
        "id": [1,2,2,4],
        "ts": pd.to_datetime(["2021-01-01","2021-01-02","2021-01-03","2021-01-04"]),
    })
    ds = Dataset(df, name="toy")
    spec = build_spec()
    spec["columns"] = {
        "age": {"dtype": "int", "min": 0, "max": 120, "nullable": True},
        "price": {"dtype": "float", "min": 0.0},
        "code": {"regex": r"^[A-Z][0-9]$"},
        "id": {"unique": True},
        "ts": {"dtype": "datetime", "monotonic": "increasing"}
    }
    out = validate(ds, spec, artifacts_dir=str(tmp_path))
    m = {mm.id: mm.value for mm in out.metrics}
    assert m["dq.validation.min.age"] <= 1.0
    assert m["dq.validation.dtype.ts"] == 1.0
    assert any(k.startswith("violations.") for k in out.artifacts.keys())

def test_composite_unique_and_cross_field(tmp_path):
    df = pd.DataFrame({
        "a": [1,1,2,2],
        "b": [1,1,2,3],
        "start": [1,2,3,4],
        "end":   [1,1,5,3]
    })
    ds = Dataset(df, name="toy2")
    spec = build_spec()
    spec["composite_unique"] = [["a","b"]]
    spec["cross_field"] = [{"expr": "start <= end", "name": "start_le_end"}]
    out = validate(ds, spec, artifacts_dir=str(tmp_path))
    m = {mm.id: mm.value for mm in out.metrics}
    assert m["dq.validation.composite_unique[a,b]"] == 0.0
    assert m["dq.validation.cross_field[start_le_end]"] < 1.0

def test_foreign_key(tmp_path):
    df = pd.DataFrame({"user_id": [1,2,3,4]})
    ref = pd.DataFrame({"id": [1,2,4]})
    ds = Dataset(df, name="orders")
    spec = build_spec()
    spec["foreign_keys"] = [{"columns": ["user_id"], "reference": ref, "ref_columns": ["id"], "name": "orders.user_id->users.id"}]
    out = validate(ds, spec, artifacts_dir=str(tmp_path))
    m = {mm.id: mm.value for mm in out.metrics}
    assert m["dq.validation.foreign_key[orders.user_id->users.id]"] == 0.75
