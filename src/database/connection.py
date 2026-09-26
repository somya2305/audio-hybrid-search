"""Postgres connection with pgvector support.

Usage:
    python -m src.database.connection    # connection + schema check
"""

import os
import sys

import psycopg
from pgvector.psycopg import register_vector

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional; plain env vars still work
    pass
else:
    load_dotenv()

TABLES = ("conversations", "transcript_chunks")


def get_connection(**kwargs):
    """Return a psycopg connection with the pgvector type registered.

    Extra keyword arguments (e.g. autocommit=True) are passed to psycopg.connect.
    """
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
    except psycopg.ProgrammingError as e:
        # register_vector fails when the extension isn't installed yet.
        sys.exit(f"Connected, but pgvector isn't enabled: {e}\nApply postgresQueries.sql first.")

    with conn:
        info = conn.info
        print(f"Connected        : {info.user}@{info.host}:{info.port}/{info.dbname}")
        print(f"Postgres         : {conn.execute('SHOW server_version').fetchone()[0]}")

        extensions = dict(conn.execute(
            "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_trgm')"
        ).fetchall())
        print(f"pgvector         : {extensions.get('vector', 'not installed')}")
        print(f"pg_trgm          : {extensions.get('pg_trgm', 'not installed')}")

        for table in TABLES:
            exists = conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0]
            if exists is None:
                print(f"{table:<17}: table missing (apply postgresQueries.sql)")
            else:
                count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                print(f"{table:<17}: {count} rows")


if __name__ == "__main__":
    main()
