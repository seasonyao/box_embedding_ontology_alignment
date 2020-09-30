import sqlite3
import time

tick = time.time()
conn = sqlite3.connect("test.db")
cur = conn.cursor()
customers_sql = """
CREATE TABLE customers (
    id integer PRIMARY KEY,
    first_name text NOT NULL,
    last_name text NOT NULL)"""
cur.execute(customers_sql)
products_sql = """
CREATE TABLE products (
    id integer PRIMARY KEY,
    name text NOT NULL,
    price real NOT NULL)"""
cur.execute(products_sql)
cur.close()
tock = time.time()
print(tock-tick)
