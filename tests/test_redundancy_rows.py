
import pandas as pd
from dqkit.types import Dataset
from dqkit.redundancy.rows import find_duplicates

def test_dup_rate_keys():
    df = pd.DataFrame({'k':[1,1,2,3]})
    rep = find_duplicates(Dataset(df), keys=['k'])
    m = {x.id:x.value for x in rep.metrics}
    assert 'dq.redundancy.rows.duplicate_rate' in m
    assert 0.0 <= m['dq.redundancy.rows.duplicate_rate'] <= 1.0
