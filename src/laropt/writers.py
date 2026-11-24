"""Result writers: persistence of the ``points`` table (one row per evaluation)
and the ``events`` table (per-event metric output) of a study."""
from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from contextlib import closing
from pathlib import Path

import pandas as pd

from .registry import register


class ResultWriter(ABC):
    name = "writer"
    ext = ""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def read(self, table: str) -> pd.DataFrame:
        """Return the table (an empty frame if it does not exist)."""

    @abstractmethod
    def write(self, table: str, df: pd.DataFrame) -> None:
        """Replace the whole table."""

    def append(self, table: str, df: pd.DataFrame, key: str = "point_id") -> None:
        """Upsert: rows whose ``key`` is in ``df`` are replaced."""
        if df is None or df.empty:
            return
        cur = self.read(table)
        if not cur.empty and key in cur.columns:
            cur = cur[~cur[key].isin(df[key].unique())]
        out = pd.concat([cur, df], ignore_index=True) if not cur.empty else df.reset_index(drop=True)
        self.write(table, out)

    def write_points(self, df): self.append("points", df)
    def write_events(self, df): self.append("events", df)
    def read_points(self): return self.read("points")
    def read_events(self): return self.read("events")


class _FileWriter(ResultWriter):
    def _file(self, table):
        return Path(f"{self.path}.{table}{self.ext}")


@register("writer", "csv")
class CsvWriter(_FileWriter):
    name, ext = "csv", ".csv"

    def read(self, table):
        f = self._file(table)
        return pd.read_csv(f) if f.exists() else pd.DataFrame()

    def write(self, table, df):
        df.to_csv(self._file(table), index=False)


@register("writer", "pickle")
class PickleWriter(_FileWriter):
    name, ext = "pickle", ".pkl"

    def read(self, table):
        f = self._file(table)
        return pd.read_pickle(f) if f.exists() else pd.DataFrame()

    def write(self, table, df):
        df.to_pickle(self._file(table))


@register("writer", "parquet")
class ParquetWriter(_FileWriter):
    name, ext = "parquet", ".parquet"

    def read(self, table):
        f = self._file(table)
        return pd.read_parquet(f) if f.exists() else pd.DataFrame()

    def write(self, table, df):
        df.to_parquet(self._file(table), index=False)


@register("writer", "sqlite")
class SQLiteWriter(ResultWriter):
    name, ext = "sqlite", ".db"

    def __init__(self, path):
        super().__init__(path)
        self.file = Path(f"{self.path}.db")

    def read(self, table):
        if not self.file.exists():
            return pd.DataFrame()
        with closing(sqlite3.connect(self.file)) as conn:
            try:
                return pd.read_sql(f"SELECT * FROM {table}", conn)
            except Exception:
                return pd.DataFrame()

    def write(self, table, df):
        with closing(sqlite3.connect(self.file)) as conn:
            df.to_sql(table, conn, if_exists="replace", index=False)

    def append(self, table, df, key="point_id"):
        if df is None or df.empty:
            return
        with closing(sqlite3.connect(self.file)) as conn:
            try:
                ids = [int(i) for i in df[key].unique()]
                conn.execute(f"DELETE FROM {table} WHERE {key} IN ({','.join('?' * len(ids))})", ids)
                df.to_sql(table, conn, if_exists="append", index=False)
                conn.commit()
                return
            except Exception:  # missing table or new columns -> rewrite the table
                conn.rollback()
        super().append(table, df, key)
