from __future__ import annotations

import sqlite3
from pathlib import Path

PREDICTION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    filename TEXT,
    model_version TEXT,
    predicted_label TEXT,
    class_probabilities_json TEXT,
    model_available INTEGER NOT NULL,
    notes TEXT
);
"""


def initialize_database(sqlite_path: Path) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(PREDICTION_TABLE_SQL)
        connection.commit()


def ping_database(sqlite_path: Path) -> None:
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute("SELECT 1")
