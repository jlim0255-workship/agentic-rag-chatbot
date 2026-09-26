"""Postgres connections."""
from __future__ import annotations

import math

import psycopg


def connect(database_url: str, timeout_s: float) -> psycopg.Connection:
    """Open a connection, failing after timeout_s instead of hanging when the DB is down."""
    return psycopg.connect(database_url, connect_timeout=max(1, math.ceil(timeout_s)))
