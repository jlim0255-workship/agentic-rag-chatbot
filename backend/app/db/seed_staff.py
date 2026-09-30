"""Create the employee_demo database and fill demo_staff with 10 FICTIONAL people for the staff-directory
tool demo (new_prd.md). Safe to re-run.

Usage (from backend/, database running):
    python -m app.db.seed_staff

Names, emails (example.com) and phone numbers (+60 3-0000 0xxx) are invented.
"""
from __future__ import annotations

import psycopg
import psycopg.conninfo

from app.config import get_settings
from app.db.connection import connect

STAFF = [  # id, name, role, topics, hours, duties
    ("e01", "Aisha Tan", "Authorised person, NCII incident notification", ["ncii_notification", "incident_response"],
     "24/7", "Notifies the Chief Executive and the NCII sector lead of cyber security incidents (PUA220 reg 2)"),
    ("e02", "Daniel Lim", "Chief Information Security Officer", ["incident_response", "management"],
     "Office hours", "Leads the incident response and decides on escalation"),
    ("e03", "Priya Nair", "Security Operations Centre lead", ["incident_response"],
     "24/7 (shift)", "Contains the incident and collects evidence"),
    ("e04", "Farid Rahman", "After-hours on-call engineer", ["incident_response"],
     "6pm-8am, weekends", "First responder outside office hours"),
    ("e05", "Mei Ling Chong", "Legal counsel", ["legal"],
     "Office hours", "Advises on legal obligations and penalties"),
    ("e06", "Rajesh Kumar", "Data protection officer", ["data_protection", "legal"],
     "Office hours", "Handles personal data breaches"),
    ("e07", "Siti Hajar Ismail", "Head of corporate communications", ["communications"],
     "Office hours", "Handles statements to staff, customers and the media"),
    ("e08", "Kevin Ong", "Chief Executive Officer", ["management"],
     "Office hours", "Approves major decisions during an incident"),
    ("e09", "Nurul Aina Yusof", "IT service desk lead", ["incident_response"],
     "8am-8pm", "Receives first reports from staff"),
    ("e10", "Grace Wong", "Compliance manager", ["ncii_notification", "legal"],
     "Office hours", "Runs the yearly risk assessment and two-yearly audit (PUA219 reg 3)"),
]


TABLE_SQL = """
CREATE TABLE IF NOT EXISTS demo_staff (
    id text PRIMARY KEY, name text NOT NULL, role text NOT NULL, topics text[] NOT NULL,
    email text NOT NULL, phone text NOT NULL, hours text NOT NULL, duties text NOT NULL
)"""


def main() -> None:
    settings = get_settings()
    # The staff directory lives in its own database (STAFF_DATABASE_URL), separate from the law index.
    # CREATE DATABASE cannot run in a transaction, so it goes through the main database with autocommit.
    dbname = psycopg.conninfo.conninfo_to_dict(settings.staff_database_url)["dbname"]
    with psycopg.connect(settings.database_url, autocommit=True) as admin:
        if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", [dbname]).fetchone():
            admin.execute(f'CREATE DATABASE "{dbname}"')
    with connect(settings.staff_database_url, settings.health_timeout_s) as conn:
        conn.execute(TABLE_SQL)
        conn.cursor().executemany(
            """INSERT INTO demo_staff (id, name, role, topics, email, phone, hours, duties)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, role = EXCLUDED.role, topics = EXCLUDED.topics,
                   email = EXCLUDED.email, phone = EXCLUDED.phone, hours = EXCLUDED.hours, duties = EXCLUDED.duties""",
            [(i, name, role, topics, f"{name.lower().replace(' ', '.')}@example.com", f"+60 3-0000 0{i[1:]}0",
              hours, duties) for i, name, role, topics, hours, duties in STAFF])
    print(f"{len(STAFF)} fictional staff in demo_staff")


if __name__ == "__main__":
    main()
