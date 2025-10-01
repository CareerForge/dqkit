
from __future__ import annotations
from typing import Any, Dict, List, Optional
import re, os
import pandas as pd
import numpy as np
from ..types import Dataset, MetricResult, RunReport

def build_spec() -> Dict[str, Any]:
    return {
        "columns": {},
        "composite_unique": [],
        "cross_field": [],
        "foreign_keys": []
    }

def validate(ds: Dataset, spec: Dict[str, Any], artifacts_dir: Optional[str]=None) -> RunReport:
    df = ds.df
    metrics: List[MetricResult] = []
    artifacts: Dict[str, str] = {}

    for col, rules in spec.get("columns", {}).items():
        if col not in df.columns:
            metrics.append(MetricResult(f"dq.validation.exists.{col}", "column", col, 0.0, unit="bool", meta={"reason":"missing_column"}))
            continue
        s = df[col]

        if "dtype" in rules:
            expected = rules["dtype"]
            ok = _check_dtype(s, expected)
            metrics.append(MetricResult(f"dq.validation.dtype.{col}", "column", col, float(ok), unit="bool", meta={"expected": expected, "actual": str(s.dtype)}))

        if "nullable" in rules:
            nullable = bool(rules["nullable"])
            null_rate = float(s.isna().mean())
            ok = (nullable or null_rate == 0.0)
            metrics.append(MetricResult(f"dq.validation.nullable.{col}", "column", col, float(ok), unit="bool", meta={"null_rate": null_rate, "allowed": nullable}))

        if "min" in rules:
            ok_ratio = float((s.dropna() >= rules["min"]).mean()) if len(s.dropna()) else 1.0
            metrics.append(MetricResult(f"dq.validation.min.{col}", "column", col, ok_ratio, unit="pass_ratio", meta={"threshold": rules["min"]}))

        if "max" in rules:
            ok_ratio = float((s.dropna() <= rules["max"]).mean()) if len(s.dropna()) else 1.0
            metrics.append(MetricResult(f"dq.validation.max.{col}", "column", col, ok_ratio, unit="pass_ratio", meta={"threshold": rules["max"]}))

        if "allowed_values" in rules:
            allowed = set(rules["allowed_values"])
            in_dom = s.isin(allowed) | s.isna()
            pass_ratio = float(in_dom.mean())
            metrics.append(MetricResult(f"dq.validation.allowed_values.{col}", "column", col, pass_ratio, unit="pass_ratio", meta={"allowed_values": list(allowed)}))
            if artifacts_dir is not None and pass_ratio < 1.0:
                path = _write_failures(artifacts_dir, ds.name, f"allowed_values_{col}", df.loc[~in_dom, [col]])
                artifacts[f"violations.allowed_values.{col}"] = path

        if "regex" in rules:
            pattern = re.compile(rules["regex"])
            mask = s.dropna().astype(str).str.match(pattern)
            aligned = s.notna()
            full_mask = pd.Series(False, index=s.index)
            full_mask.loc[aligned.index[aligned]] = mask
            pass_ratio = float(full_mask.mean() if len(full_mask)>0 else 1.0)
            metrics.append(MetricResult(f"dq.validation.regex.{col}", "column", col, pass_ratio, unit="pass_ratio", meta={"pattern": pattern.pattern}))
            if artifacts_dir is not None and pass_ratio < 1.0:
                path = _write_failures(artifacts_dir, ds.name, f"regex_{col}", df.loc[~full_mask, [col]])
                artifacts[f"violations.regex.{col}"] = path

        if "monotonic" in rules:
            mode = str(rules["monotonic"])
            pass_bool = _check_monotonic(s, mode)
            metrics.append(MetricResult(f"dq.validation.monotonic.{col}", "column", col, float(pass_bool), unit="bool", meta={"mode": mode}))

        if rules.get("unique"):
            unique_bool = bool(s.isna().sum()==0 and s.nunique(dropna=False) == len(s))
            metrics.append(MetricResult(f"dq.validation.unique.{col}", "column", col, float(unique_bool), unit="bool"))

    for idx, cols in enumerate(spec.get("composite_unique", []) or []):
        name = ",".join(cols)
        dup_groups = df.duplicated(subset=cols, keep=False)
        unique_bool = bool((~dup_groups).all())
        metrics.append(MetricResult(f"dq.validation.composite_unique[{name}]", "dataset", cols, float(unique_bool), unit="bool"))
        if artifacts_dir is not None and not unique_bool:
            path = _write_failures(artifacts_dir, ds.name, f"composite_unique_{idx}", df.loc[dup_groups, cols])
            artifacts[f"violations.composite_unique[{name}]"] = path

    for rule in (spec.get("cross_field", []) or []):
        expr = rule["expr"]
        name = rule.get("name", expr)
        try:
            mask = pd.eval(expr, engine="python", parser="pandas", local_dict={}, global_dict={}, target=df)
            if not isinstance(mask, (pd.Series, np.ndarray)):
                pass_ratio = 1.0 if bool(mask) else 0.0
                failing = df.index if pass_ratio == 0.0 else df.index[:0]
            else:
                mask = pd.Series(mask, index=df.index).fillna(False)
                pass_ratio = float(mask.mean())
                failing = df.index[~mask]
        except Exception:
            pass_ratio = 0.0
            failing = df.index
        metrics.append(MetricResult(f"dq.validation.cross_field[{name}]", "dataset", name, pass_ratio, unit="pass_ratio", meta={"expr": expr}))
        if artifacts_dir is not None and pass_ratio < 1.0:
            path = _write_failures(artifacts_dir, ds.name, f"cross_field_{_slug(name)}", df.loc[failing])
            artifacts[f"violations.cross_field[{name}]"] = path

    for fk in (spec.get("foreign_keys", []) or []):
        cols = fk["columns"]
        ref_df = fk["reference"]
        ref_cols = fk.get("ref_columns", cols)
        name = fk.get("name", f"fk({','.join(cols)})->ref({','.join(ref_cols)})")
        merged = df[cols].merge(ref_df[ref_cols].drop_duplicates(), left_on=cols, right_on=ref_cols, how="left", indicator=True)
        ok_mask = merged["_merge"] == "both"
        pass_ratio = float(ok_mask.mean())
        metrics.append(MetricResult(f"dq.validation.foreign_key[{name}]", "dataset", name, pass_ratio, unit="pass_ratio"))
        if artifacts_dir is not None and pass_ratio < 1.0:
            bad_rows = df.loc[~ok_mask]
            path = _write_failures(artifacts_dir, ds.name, f"foreign_key_{_slug(name)}", bad_rows[cols])
            artifacts[f"violations.foreign_key[{name}]"] = path

    return RunReport(metrics=metrics, artifacts=artifacts, meta={"dataset": ds.name})

def _check_dtype(s: pd.Series, expected: str) -> bool:
    kind = expected.lower()
    if kind in ("int", "integer"):
        return pd.api.types.is_integer_dtype(s.dropna())
    if kind in ("float", "number", "numeric"):
        return pd.api.types.is_float_dtype(s.dropna()) or pd.api.types.is_integer_dtype(s.dropna())
    if kind in ("bool", "boolean"):
        return pd.api.types.is_bool_dtype(s.dropna())
    if kind in ("str", "string", "object"):
        return pd.api.types.is_object_dtype(s.dropna()) or pd.api.types.is_string_dtype(s.dropna())
    if kind in ("datetime", "datetime64"):
        return pd.api.types.is_datetime64_any_dtype(s.dropna())
    return False

def _check_monotonic(s: pd.Series, mode: str) -> bool:
    x = s.dropna().to_numpy()
    if len(x) <= 1:
        return True
    if mode == "increasing":
        return bool(np.all(np.diff(x) >= 0))
    if mode == "strict_increasing":
        return bool(np.all(np.diff(x) > 0))
    if mode == "decreasing":
        return bool(np.all(np.diff(x) <= 0))
    if mode == "strict_decreasing":
        return bool(np.all(np.diff(x) < 0))
    return False

def _write_failures(artifacts_dir: str, ds_name: str, slug: str, frame: pd.DataFrame) -> str:
    os.makedirs(artifacts_dir, exist_ok=True)
    filename = f"{_slug(ds_name)}__{slug}.csv"
    path = os.path.join(artifacts_dir, filename)
    frame.to_csv(path, index=False)
    return path

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s)).strip("_")
