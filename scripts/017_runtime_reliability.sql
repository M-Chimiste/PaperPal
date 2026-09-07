CREATE TABLE IF NOT EXISTS task_dispatch (
    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
    handler TEXT NOT NULL,
    queue TEXT NOT NULL DEFAULT 'general',
    status TEXT NOT NULL DEFAULT 'pending',
    owner TEXT,
    lease_until TIMESTAMPTZ,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS task_dispatch_claim ON task_dispatch(queue, status, lease_until);

-- Inserting a supported task also records dispatch in the same transaction.
-- This closes the crash window between HTTP task creation and enqueue_task.
CREATE OR REPLACE FUNCTION enqueue_supported_task() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE handler_name TEXT;
BEGIN
    handler_name := CASE
        WHEN NEW.task_type IN ('newsletter','podcast','visualizer','database_export','database_import',
            'mindmap_expand','mindmap_pdf_parse','profile_aware_ingest','bulk_embed',
            'custom_newsletter','profile_newsletter','star-map') THEN NEW.task_type
        WHEN NEW.task_type IN ('research-agent-single','research-agent-multi') THEN 'research'
        WHEN NEW.task_type = 'profile_interest_clustering' THEN 'profile_interest'
        ELSE NULL END;
    IF handler_name IS NOT NULL AND NEW.status = 'pending' THEN
        INSERT INTO task_dispatch(task_id,handler,queue)
        VALUES(NEW.task_id,handler_name,CASE WHEN handler_name='visualizer' THEN 'visualizer' ELSE 'general' END)
        ON CONFLICT(task_id) DO NOTHING;
    END IF;
    RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS tasks_durable_dispatch ON tasks;
CREATE TRIGGER tasks_durable_dispatch AFTER INSERT ON tasks
    FOR EACH ROW EXECUTE FUNCTION enqueue_supported_task();

CREATE TABLE IF NOT EXISTS delivery_receipts (
    delivery_key TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('sending', 'sent', 'uncertain', 'retry')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS task_stage_events (
    id BIGSERIAL PRIMARY KEY,
    task_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms DOUBLE PRECISION,
    error_type TEXT,
    details JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS task_stage_events_task ON task_stage_events(task_id, created_at);

-- Model identity is filtered by retrieval; dimensions remain the existing 768.
CREATE INDEX IF NOT EXISTS idx_papers_embedding_hnsw
    ON papers USING hnsw (embedding vector_cosine_ops);
