"""
EMIPredict AI - Database / CRUD Module
SQLite-backed storage for customer financial profiles, used by the Admin page.
Kept intentionally simple (single table, no ORM) since the project only
needs basic CRUD, not a production banking schema.
"""

import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "data/emi_customers.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    marital_status TEXT,
    education TEXT,
    monthly_salary REAL,
    employment_type TEXT,
    years_of_employment REAL,
    company_type TEXT,
    house_type TEXT,
    monthly_rent REAL,
    family_size INTEGER,
    dependents INTEGER,
    school_fees REAL,
    college_fees REAL,
    travel_expenses REAL,
    groceries_utilities REAL,
    other_monthly_expenses REAL,
    existing_loans TEXT,
    current_emi_amount REAL,
    credit_score REAL,
    bank_balance REAL,
    emergency_fund REAL,
    emi_scenario TEXT,
    requested_amount REAL,
    requested_tenure INTEGER,
    eligibility_result TEXT,
    max_safe_emi_result REAL,
    created_at TEXT
);
"""


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(SCHEMA)
    conn.commit()
    conn.close()


def create_customer(data: dict) -> int:
    """Inserts a new customer record. data keys must match the schema columns
    (excluding id/created_at). Returns the new row's id."""
    conn = get_connection()
    data = {**data, "created_at": datetime.now().isoformat(timespec="seconds")}
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    cur = conn.execute(f"INSERT INTO customers ({cols}) VALUES ({placeholders})", list(data.values()))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def read_all_customers() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM customers ORDER BY id DESC", conn)
    conn.close()
    return df


def read_customer(customer_id: int) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_customer(customer_id: int, data: dict) -> bool:
    """Updates the given fields for a customer. Returns True if a row was updated."""
    conn = get_connection()
    set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
    values = list(data.values()) + [customer_id]
    cur = conn.execute(f"UPDATE customers SET {set_clause} WHERE id = ?", values)
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def delete_customer(customer_id: int) -> bool:
    conn = get_connection()
    cur = conn.execute("DELETE FROM customers WHERE id = ?", (customer_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


if __name__ == "__main__":
    # smoke test
    init_db()
    new_id = create_customer({
        "name": "Test Customer", "age": 30, "gender": "Male", "marital_status": "Single",
        "education": "Graduate", "monthly_salary": 50000, "employment_type": "Private",
        "years_of_employment": 3, "company_type": "MNC", "house_type": "Rented",
        "monthly_rent": 12000, "family_size": 1, "dependents": 0, "school_fees": 0,
        "college_fees": 0, "travel_expenses": 2000, "groceries_utilities": 6000,
        "other_monthly_expenses": 1000, "existing_loans": "No", "current_emi_amount": 0,
        "credit_score": 720, "bank_balance": 80000, "emergency_fund": 50000,
        "emi_scenario": "Personal Loan EMI", "requested_amount": 200000, "requested_tenure": 24,
        "eligibility_result": "Eligible", "max_safe_emi_result": 15000,
    })
    print(f"Created customer id={new_id}")
    print(read_customer(new_id))
    print(f"Update success: {update_customer(new_id, {'age': 31})}")
    print(f"Total customers: {len(read_all_customers())}")
    print(f"Delete success: {delete_customer(new_id)}")
