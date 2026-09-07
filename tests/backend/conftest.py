"""Integration fixtures recreate only the expendable *_test database."""
import os
import json
import pathlib
TEST_DB_URL = os.environ["DATABASE_URL"]
TEST_DB_NAME = TEST_DB_URL.rsplit("/", 1)[-1]

import psycopg
import pytest

ADMIN_URL = TEST_DB_URL.rsplit("/", 1)[0] + "/postgres"
GOLDEN_DIR = pathlib.Path(__file__).parents[1] / "goldens"

# A deterministic 768-dim embedding pointing along axis 0 / axis 1.
EMB_DIM = 768


def axis_embedding(axis: int, value: float = 1.0) -> list[float]:
    vec = [0.0] * EMB_DIM
    vec[axis] = value
    return vec


def pgvector_literal(vec: list[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


@pytest.fixture(scope="session", autouse=True)
def migrated_db():
    """Drop/recreate the test DB and run the production migration chain.

    Running MigrationRunner against an empty database every session IS the
    migration characterization test — it exercises the exact startup path.
    """
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=5)
    except Exception as exc:  # pragma: no cover
        pytest.exit(
            f"Test database not reachable at {ADMIN_URL}: {exc}\n"
            "Start it with: make test-db",
            returncode=2,
        )
    try:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DB_NAME}")
    finally:
        admin.close()

    # First theseus_insight import happens here, with a healthy empty DB.
    from theseus_insight.db.migrations import MigrationRunner

    runner = MigrationRunner()
    applied, skipped, issues = runner.run_migrations()
    assert issues == [], f"Migration issues on fresh database: {issues}"
    yield {"applied": applied, "skipped": skipped}


@pytest.fixture(scope="session")
def client(migrated_db):
    """FastAPI TestClient with lifespan deliberately NOT started.

    See module docstring point 3 — never wrap this in `with`.
    """
    from fastapi.testclient import TestClient
    from theseus_insight.main import app

    return TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))


@pytest.fixture()
def db():
    """Raw psycopg connection for seeding/inspecting, independent of app pools."""
    conn = psycopg.connect(TEST_DB_URL, autocommit=True, row_factory=psycopg.rows.dict_row)
    yield conn
    conn.close()


def _truncate(conn) -> None:
    conn.execute(
        "TRUNCATE papers, logs, tasks RESTART IDENTITY CASCADE"
    )
    conn.execute("DELETE FROM profile_research_interests")
    conn.execute("DELETE FROM paper_profile_scores")
    conn.execute("DELETE FROM research_profiles WHERE is_default = FALSE")


@pytest.fixture()
def empty_db(db):
    """Tables cleared (migration-created Default profile kept)."""
    _truncate(db)
    return db


@pytest.fixture()
def seeded_data(db):
    """Deterministic fixture dataset, inserted with raw SQL.

    Raw SQL on purpose: characterization tests must not depend on the
    repository code that the refactor is about to move.
    """
    _truncate(db)

    papers = [
        # (title, abstract, date, date_run, score, rationale, related, url, embedding)
        ("Alpha Paper", "Transformers for everything.", "2025-01-10", "2025-01-11",
         9.0, "highly relevant", True, "https://example.org/alpha",
         pgvector_literal(axis_embedding(0))),
        ("Beta Paper", "Graph methods for citation analysis.", "2025-01-05", "2025-01-06",
         5.0, "somewhat relevant", False, "https://example.org/beta",
         pgvector_literal(axis_embedding(1))),
        ("Gamma Paper", "Survey of obsolete techniques.", "2025-01-01", "2025-01-02",
         2.0, "not relevant", False, "https://example.org/gamma", None),
    ]
    for row in papers:
        db.execute(
            """
            INSERT INTO papers (title, abstract, date, date_run, score, rationale,
                                related, url, embedding, embedding_model)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'fake-model')
            """,
            row,
        )

    default_profile = db.execute(
        "SELECT id FROM research_profiles WHERE is_default = TRUE"
    ).fetchone()
    test_profile = db.execute(
        """
        INSERT INTO research_profiles (name, description, color, tags, is_active, is_default)
        VALUES ('Test Profile', 'fixture profile', '#ff0000', '["ml"]'::jsonb, TRUE, FALSE)
        RETURNING id
        """
    ).fetchone()

    db.execute(
        """
        INSERT INTO paper_profile_scores (paper_id, profile_id, score, related, rationale, judge_model)
        VALUES (1, %(pid)s, 8, TRUE, 'fixture: on-topic', 'fake-judge'),
               (2, %(pid)s, 3, FALSE, 'fixture: off-topic', 'fake-judge')
        """,
        {"pid": test_profile["id"]},
    )

    return {
        "paper_ids": [1, 2, 3],
        "default_profile_id": default_profile["id"] if default_profile else None,
        "test_profile_id": test_profile["id"],
    }


