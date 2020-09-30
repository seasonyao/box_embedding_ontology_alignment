import sqlite3
from sqlite3 import Connection
from .learner import Recorder
from .utils import get_timestamp
from typing import List

PRINT_TIME = True

def write_dict_(conn: Connection, table_name: str, d: dict, print_time: bool = PRINT_TIME):
    try:
        # Convert tensors to values we can write to SQL
        for k, v in d.items():
            if hasattr(v, 'dtype'):
                d[k] = v.item()

        if print_time:
            print(f"{get_timestamp()}: SQL start writing rows to {table_name}")
        c = conn.cursor()
        c.execute(f"insert into {table_name} ({','.join(d.keys())}) values ({':' +', :'.join(d.keys())})", d)
        conn.commit()
        c.close()
        if print_time:
            print(f"{get_timestamp()}: SQL end writing rows to {table_name}")
        return c.lastrowid # this is guaranteed to be the ID generated by the previous operation
    except sqlite3.Error as e:
        if print_time:
            print(f"{get_timestamp()}: {e}")
        raise e

def create_or_update_table_and_cols_(conn: Connection, table_name: str, cols: List[str], print_time: bool = PRINT_TIME):
    try:
        if print_time:
            print(f"{get_timestamp()}: SQL start writing table cols to {table_name}")
        c = conn.cursor()
        c.execute(f"create table if not exists {table_name} ({','.join(cols)})")
        for col in cols:
            try:
                c.execute(f"alter table {table_name} add column {col}")
            except sqlite3.Error as e:
                if print_time:
                    print(f"{get_timestamp()}: {e}")
        conn.commit()
        c.close()
        if print_time:
            print(f"{get_timestamp()}: SQL end writing table cols to {table_name}")
    except sqlite3.Error as e:
        if print_time:
            print(f"{get_timestamp()}: {e}")
        raise e

def save_recorder_to_sql_(conn: Connection, rec: Recorder, table_name: str, training_instance_id: int, print_time: bool = PRINT_TIME):
    cols = [
        "[id] INTEGER PRIMARY KEY",
        "[training_instance_id] INTEGER NOT NULL",
        "[epochs] REAL", # Note: we add this explicitly because it is actually the index label of the dataframe
    ]
    cols += [f"[{col}] REAL" for col in rec._data.columns]
    cols += ["FOREIGN KEY(training_instance_id) REFERENCES Training_Instances(id)"]
    create_or_update_table_and_cols_(conn, table_name, cols)

    if print_time:
        print(f"{get_timestamp()}: SQL start writing dataframe to {table_name}")
    try:
        rec.dataframe["training_instance_id"] = training_instance_id
        rec.dataframe.to_sql(table_name, conn, if_exists="append", index_label="epochs")
    except sqlite3.Error as e:
        if print_time:
            print(f"{get_timestamp()}: {e}")
        raise e
    if print_time:
        print(f"{get_timestamp()}: SQL end writing dataframe to {table_name}")
