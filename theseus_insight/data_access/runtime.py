"""Durable dispatch; session locks prevent concurrent execution after lease expiry."""
from contextlib import contextmanager
import json
from ..db import get_cursor, get_connection


class DispatchRepository:
    @staticmethod
    def enqueue(task_id, handler, queue):
        with get_cursor() as cur:
            cur.execute("""INSERT INTO task_dispatch(task_id, handler, queue)
                VALUES (%s,%s,%s) ON CONFLICT(task_id) DO NOTHING""", (task_id, handler, queue))

    @staticmethod
    def candidates(queue):
        with get_cursor() as cur:
            cur.execute("""SELECT task_id FROM task_dispatch WHERE queue=%s AND
                (status='pending' OR (status='leased' AND lease_until < now()))
                ORDER BY created_at LIMIT 20""", (queue,))
            return [row['task_id'] for row in cur.fetchall()]

    @staticmethod
    @contextmanager
    def claim(task_id, owner):
        # Keep a dedicated connection (and lock) for the entire handler lifetime.
        # A lease expiry alone never allows concurrent execution.
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS acquired", ('task:' + task_id,))
                acquired = cur.fetchone()['acquired']
                if not acquired:
                    yield None
                    return
                try:
                    cur.execute("""UPDATE task_dispatch SET status='leased', owner=%s,
                        lease_until=now()+interval '60 seconds', attempts=attempts+1, updated_at=now()
                        WHERE task_id=%s AND (status='pending' OR (status='leased' AND lease_until < now()))
                        RETURNING *""", (owner, task_id))
                    row = cur.fetchone()
                    conn.commit()
                    yield row
                finally:
                    cur.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", ('task:' + task_id,))
                    conn.commit()

    @staticmethod
    def renew(task_id, owner):
        with get_cursor() as cur:
            cur.execute("""UPDATE task_dispatch SET lease_until=now()+interval '60 seconds', updated_at=now()
                WHERE task_id=%s AND owner=%s AND status='leased'""", (task_id, owner))
            if cur.rowcount != 1:
                raise RuntimeError('Task lease ownership lost')

    @staticmethod
    def finish(task_id, owner):
        with get_cursor() as cur:
            cur.execute("""UPDATE task_dispatch SET status='done', lease_until=NULL, updated_at=now()
                WHERE task_id=%s AND owner=%s""", (task_id, owner))

    @staticmethod
    def retry(task_id):
        with get_cursor() as cur:
            cur.execute("""UPDATE task_dispatch SET status='pending', owner=NULL, lease_until=NULL, attempts=0
                WHERE task_id=%s AND status='done' RETURNING task_id""", (task_id,))
            if not cur.fetchone():
                raise ValueError('Task is active or has no durable handler')
            cur.execute("UPDATE tasks SET status='pending', error=NULL, end_time=NULL WHERE task_id=%s", (task_id,))


class DeliveryRepository:
    @staticmethod
    def begin(key, task_id):
        with get_cursor() as cur:
            cur.execute("""INSERT INTO delivery_receipts(delivery_key, task_id, status)
                VALUES(%s,%s,'sending') ON CONFLICT(delivery_key) DO UPDATE
                SET status='sending', updated_at=now() WHERE delivery_receipts.status='retry'
                RETURNING status""", (key, task_id))
            if cur.fetchone():
                return True
            cur.execute("SELECT status FROM delivery_receipts WHERE delivery_key=%s", (key,))
            if cur.fetchone()['status'] == 'sent':
                return False
            raise RuntimeError('Email delivery is uncertain; review delivery receipts before explicitly retrying')

    @staticmethod
    def finish(key, status):
        with get_cursor() as cur:
            cur.execute("UPDATE delivery_receipts SET status=%s, updated_at=now() WHERE delivery_key=%s", (status, key))


def record_event(task_id, stage, status, duration_ms=None, error_type=None, details=None):
    with get_cursor() as cur:
        cur.execute("""INSERT INTO task_stage_events(task_id,stage,status,duration_ms,error_type,details)
            VALUES(%s,%s,%s,%s,%s,%s)""", (task_id, stage, status, duration_ms, error_type, json.dumps(details or {})))
