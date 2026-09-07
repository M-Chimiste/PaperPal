"""Characterize the migration system — the production startup path."""


def test_fresh_db_migrations_apply_cleanly(migrated_db, db):
    rows = db.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
    versions = [r["version"] for r in rows]
    assert versions == list(range(19)), f"Expected migrations 0-18 applied, got {versions}"

    from theseus_insight.db.migrations import MigrationRunner

    missing = MigrationRunner()._verify_critical_tables()
    assert missing == [], f"Critical tables missing after migration: {missing}"


def test_migrations_idempotent(migrated_db):
    from theseus_insight.db.migrations import MigrationRunner

    applied, skipped, issues = MigrationRunner().run_migrations()
    assert applied == 0
    assert skipped == 19
    assert issues == []


def test_staging_tables_exist(db):
    rows = db.execute(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name IN
              ('papers_staging', 'embeddings_staging', 'keywords_staging',
               'paper_profile_scores_staging')
        """
    ).fetchall()
    assert {r["table_name"] for r in rows} == {
        "papers_staging",
        "embeddings_staging",
        "keywords_staging",
        "paper_profile_scores_staging",
    }


def test_default_profile_created_on_fresh_db(db):
    """Migration 002 creates a Default profile on an empty database."""
    row = db.execute(
        "SELECT name, is_active, is_default FROM research_profiles WHERE is_default = TRUE"
    ).fetchone()
    assert row is not None
    assert row["name"] == "Default"
    assert row["is_active"] is True
