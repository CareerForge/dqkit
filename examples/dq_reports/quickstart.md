# Data Quality Report
_Generated: 2025-09-30 19:29:57 UTC_

## Profiling
| Metric ID | Level | Target | Value | Unit |
|---|---|---|---:|---|
| `dq.profile.count.age` | column | age | 5 |  |
| `dq.profile.missing_rate.age` | column | age | 0.16666666666666666 |  |
| `dq.profile.distinct.age` | column | age | 5 |  |
| `dq.profile.dtype.age` | column | age | float64 |  |
| `dq.profile.min.age` | column | age | 20.0 |  |
| `dq.profile.max.age` | column | age | 50.0 |  |
| `dq.profile.mean.age` | column | age | 34.0 |  |
| `dq.profile.std.age` | column | age | 12.942179105544785 |  |
| `dq.profile.q5.age` | column | age | 21.0 |  |
| `dq.profile.q25.age` | column | age | 25.0 |  |
| `dq.profile.q50.age` | column | age | 30.0 |  |
| `dq.profile.q75.age` | column | age | 45.0 |  |
| `dq.profile.q95.age` | column | age | 49.0 |  |
| `dq.profile.count.fare` | column | fare | 6 |  |
| `dq.profile.missing_rate.fare` | column | fare | 0.0 |  |
| `dq.profile.distinct.fare` | column | fare | 6 |  |
| `dq.profile.dtype.fare` | column | fare | float64 |  |
| `dq.profile.min.fare` | column | fare | 55.0 |  |
| `dq.profile.max.fare` | column | fare | 200.0 |  |
| `dq.profile.mean.fare` | column | fare | 109.26666666666667 |  |
| `dq.profile.std.fare` | column | fare | 49.59450238349677 |  |
| `dq.profile.q5.fare` | column | fare | 61.375 |  |
| `dq.profile.q25.fare` | column | fare | 85.35 |  |
| `dq.profile.q50.fare` | column | fare | 99.95 |  |
| `dq.profile.q75.fare` | column | fare | 115.15 |  |
| `dq.profile.q95.fare` | column | fare | 180.05 |  |
| `dq.profile.count.label` | column | label | 6 |  |
| `dq.profile.missing_rate.label` | column | label | 0.0 |  |
| `dq.profile.distinct.label` | column | label | 2 |  |
| `dq.profile.dtype.label` | column | label | int64 |  |
| `dq.profile.min.label` | column | label | 0.0 |  |
| `dq.profile.max.label` | column | label | 1.0 |  |
| `dq.profile.mean.label` | column | label | 0.6666666666666666 |  |
| `dq.profile.std.label` | column | label | 0.5163977794943223 |  |
| `dq.profile.q5.label` | column | label | 0.0 |  |
| `dq.profile.q25.label` | column | label | 0.25 |  |
| `dq.profile.q50.label` | column | label | 1.0 |  |
| `dq.profile.q75.label` | column | label | 1.0 |  |
| `dq.profile.q95.label` | column | label | 1.0 |  |
| `dq.profile.n_rows` | dataset | * | 6 |  |
| `dq.profile.n_cols` | dataset | * | 3 |  |

## Missingness
| Metric ID | Level | Target | Value | Unit |
|---|---|---|---:|---|
| `dq.missing.row_rate` | dataset | * | 0.16666666666666666 |  |
| `dq.missing.rate.age` | column | age | 0.16666666666666666 |  |
| `dq.missing.rate.fare` | column | fare | 0.0 |  |
| `dq.missing.rate.label` | column | label | 0.0 |  |
| `dq.missing.cooccur.age.fare` | dataset | ['age', 'fare'] | 0.0 |  |
| `dq.missing.cooccur.age.label` | dataset | ['age', 'label'] | 0.0 |  |
| `dq.missing.cooccur.fare.label` | dataset | ['fare', 'label'] | 0.0 |  |
| `dq.missing.top_patterns` | dataset | ['age', 'fare', 'label'] | [{"pattern": "000", "count": 5}, {"pattern": "100", "count": 1}] |  |

## Imbalance
| Metric ID | Level | Target | Value | Unit |
|---|---|---|---:|---|
| `dq.imbalance.counts` | dataset | label | {"1": 4, "0": 2} |  |
| `dq.imbalance.ir` | dataset | label | 2.0 |  |
| `dq.imbalance.effective_n` | dataset | label | 5.99300399899993 |  |
| `dq.imbalance.gini` | dataset | label | 0.4444444444444444 |  |
| `dq.imbalance.rarity.1` | dataset | 1 | 0.0 |  |
| `dq.imbalance.rarity.0` | dataset | 0 | 1.0 |  |

## Artifacts
- `artifact.hist.age`: artifacts/age_hist.csv
- `artifact.hist.fare`: artifacts/fare_hist.csv
- `artifact.hist.label`: artifacts/label_hist.csv
- `artifact.missing.cooccurrence`: artifacts/missing_cooccurrence.csv
- `artifact.missing.top_patterns`: artifacts/missing_top_patterns.csv