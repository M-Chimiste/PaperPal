CREATE TABLE IF NOT EXISTS newsletter_editions (
    task_id TEXT PRIMARY KEY,
    newsletter_id INTEGER NOT NULL REFERENCES newsletters(id) ON DELETE CASCADE,
    profile_ids INTEGER[] NOT NULL DEFAULT '{}',
    paper_keys TEXT[] NOT NULL DEFAULT '{}',
    artifact JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS newsletter_editions_created ON newsletter_editions(created_at);
