import sqlite3
from sqlite3 import Connection
from .learner import Recorder
from .utils import get_timestamp
from typing import List
import contextlib
import time
from .logger import Logger
from dataclasses import dataclass

# Excessive try/catch and printing is because SQL writes on NFS were locking


@dataclass
class SQLiteLogger(Logger):
    conn: Connection
    print_time: bool = False

    def _write_row(self, category: str, row: dict):
        table_name = category
        # Convert tensors to values we can write to SQL
        for k, v in row.items():
            if hasattr(v, 'dtype'):
                row[k] = v.item()
        try:
            with self.conn:
                with contextlib.closing(self.conn.cursor()) as c:
                    if self.print_time:
                        print(f"{get_timestamp()}: SQL start writing rows to {table_name}")
                    c.execute(f"insert into {table_name} ({','.join(row.keys())}) values ({':' +', :'.join(row.keys())})", row)
                    row_id = c.lastrowid
            if self.print_time:
                print(f"{get_timestamp()}: SQL end writing rows to {table_name}")
            return row_id
        except sqlite3.Error as e:
            if self.print_time:
                print(f"{get_timestamp()}: {e}")
            raise e

    def create_or_update_table_and_cols_(self, table_name: str, cols: List[str]):
        try:
            if self.print_time:
                print(f"{get_timestamp()}: SQL start writing table cols to {table_name}")
            with self.conn:
                with contextlib.closing(self.conn.cursor()) as c:
                    c.execute(f"create table if not exists {table_name} ({','.join(cols)})")
                    for col in cols:
                        try:
                            c.execute(f"alter table {table_name} add column {col}")
                        except sqlite3.Error as e:
                            if self.print_time:
                                print(f"{get_timestamp()}: {e}")
            if self.print_time:
                print(f"{get_timestamp()}: SQL end writing table cols to {table_name}")
        except sqlite3.Error as e:
            if self.print_time:
                print(f"{get_timestamp()}: {e}")
            raise e

    def save_recorder_to_sql_(self, rec: Recorder, table_name: str, training_instance_id: int):
        cols = [
            "[id] INTEGER PRIMARY KEY",
            "[training_instance_id] INTEGER NOT NULL",
            "[epochs] REAL", # Note: we add this explicitly because it is actually the index label of the dataframe
        ]
        cols += [f"[{col}] REAL" for col in rec.dataframe().columns]
        cols += ["FOREIGN KEY(training_instance_id) REFERENCES Training_Instances(id)"]
        self.create_or_update_table_and_cols_(table_name, cols)

        if self.print_time:
            print(f"{get_timestamp()}: SQL start writing dataframe to {table_name}")
        try:
            rec.dataframe()["training_instance_id"] = training_instance_id
            rec.dataframe().to_sql(table_name, self.conn, if_exists="append", index_label="epochs")
        except sqlite3.Error as e:
            if self.print_time:
                print(f"{get_timestamp()}: {e}")
            raise e
        if self.print_time:
            print(f"{get_timestamp()}: SQL end writing dataframe to {table_name}")
