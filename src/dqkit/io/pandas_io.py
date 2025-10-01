
import pandas as pd
from ..types import Dataset

def from_csv(path: str, **kwargs) -> Dataset:
    return Dataset(pd.read_csv(path, **kwargs), name=path)

def from_dataframe(df: pd.DataFrame, name: str = "df") -> Dataset:
    return Dataset(df.copy(), name=name)
