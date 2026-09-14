
import sqlite3

from typing import List, Dict
from laropt.core.parameters import ParamSpec, ResultSpec, ParamSet


class RunDB:

    def __init__(self, db_path: str, specs: Dict[str, ParamSpec], result_specs: List[ResultSpec]):
        self.conn = sqlite3.connect(db_path)
        self.specs = specs
        self.result_specs = result_specs
        self._create_table()

    def _create_table(self):
        cursor = self.conn.cursor()

        columns = [
            "id INTEGER PRIMARY KEY AUTOINCREMENT",
            "jobNum INTEGER",
            "timestamp TEXT",
            "fcl_filename TEXT",
            "output_filename TEXT",
            "hist_filename TEXT",
            "notes TEXT"
        ]

        # -------------------------
        # Parameters
        # -------------------------
        for name, spec in self.specs.items():
            dtype = "INTEGER" if spec.kind == "int" else "REAL"

            if spec.size == 1:
                columns.append(f"{name} {dtype}")
            else:
                for i in range(spec.size):
                    columns.append(f"{name}_{i} {dtype}")

        # -------------------------
        # Results (NEW)
        # -------------------------
        for rs in self.result_specs:
            columns.append(f"{rs.name} {rs.dtype}")

        cursor.execute(f"CREATE TABLE IF NOT EXISTS runs ({', '.join(columns)})")
        self.conn.commit()

    def insert(self, params: ParamSet, meta: dict, results: dict):
        cursor = self.conn.cursor()

        data = {**meta, **params.to_flat_dict(), **results}
        keys = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))

        cursor.execute(
            f'INSERT INTO runs ({keys}) VALUES ({placeholders})',
            tuple(data.values())
        )

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

    def rollback(self):
        self.conn.rollback()