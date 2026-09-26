"""Postgres connection with pgvector support.

Usage:
    python -m src.database.connection    # check the connection and schema
"""

import os
import sys

import psycopg
from pgvector.psycopg import register_vector

import src.common  # noqa: F401  (loads DB_* settings from .env)

TABLES = ("conversations", "transcript_chunks")


def get_connection(**kwargs):
    """psycopg connection with the pgvector type registered; kwargs go to psycopg.connect."""
    conn = psycopg.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "5432")),
        dbname=os.environ.get("DB_NAME", "audio_search"),
        user=os.environ.get("DB_USER", "rag_user"),
        password=os.environ.get("DB_PASSWORD", "rag_password"),
        **kwargs,
    )
    register_vector(conn)
    return conn


def main():
    try:
        conn = get_connection()
    except psycopg.OperationalError as e:
        sys.exit(f"Could not connect to Postgres: {e}\nIs the container running? (docker compose up -d)")
    except psycopg.ProgrammingError as e:  # register_vector: extension not installed
        sys.exit(f"Connected, but pgvector isn't enabled: {e}\nApply db/schema.sql first.")

    with conn:
        info = conn.info
        extensions = dict(conn.execute(
            "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_trgm')"
        ).fetchall())
        print(f"Connected        : {info.user}@{info.host}:{info.port}/{info.dbname}")
        print(f"Postgres         : {conn.execute('SHOW server_version').fetchone()[0]}")
        print(f"pgvector         : {extensions.get('vector', 'not installed')}")
        print(f"pg_trgm          : {extensions.get('pg_trgm', 'not installed')}")
        for table in TABLES:
            if conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None:
                print(f"{table:<17}: missing (apply db/schema.sql)")
            else:
                print(f"{table:<17}: {conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0]} rows")


if __name__ == "__main__":
    main()
