"""Repeatable synthetic search benchmark, restricted to an expendable *_test DB."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=10000)
    parser.add_argument('--output', type=Path, default=Path('data/evaluation/search-benchmark.json'))
    args = parser.parse_args()
    if not 100 <= args.rows <= 300000:
        parser.error('--rows must be between 100 and 300000')
    dsn = os.environ.get('TEST_DATABASE_URL', 'postgresql://theseus:theseus@localhost:5434/theseus_benchmark_test')
    if not dsn.rsplit('/', 1)[-1].endswith('_test'):
        parser.error('Benchmark refuses databases without the _test suffix')
    import psycopg
    from psycopg import sql
    database = dsn.rsplit('/', 1)[-1]
    with psycopg.connect(dsn.rsplit('/', 1)[0] + '/postgres', autocommit=True) as admin:
        if not admin.execute('SELECT 1 FROM pg_database WHERE datname=%s', (database,)).fetchone():
            admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(database)))
    os.environ['DATABASE_URL'] = dsn
    from theseus_insight.db.migrations import MigrationRunner
    _, _, issues = MigrationRunner().run_migrations()
    if issues:
        raise RuntimeError('Benchmark database migrations failed')
    from theseus_insight.db import get_cursor
    from theseus_insight.data_access import PaperRepository
    tag = 'benchmark-' + str(uuid.uuid4())
    class Model:
        def invoke(self, query): return [1.] + [0.] * 766 + [1.]
    try:
        with get_cursor() as cur:
            cur.execute("""INSERT INTO papers(title,abstract,date,date_run,score,rationale,related,url,embedding,embedding_model)
                SELECT 'Retrieval paper '||i, CASE WHEN i %% 10=0 THEN 'retrieval augmented generation' ELSE 'graph classification' END,
                    '2026-01-01','2026-01-01',5,'synthetic',true,%s||i,
                    ('['||(i %% 10)::text||','||repeat('0,',766)||'1]')::vector,%s
                FROM generate_series(1,%s) i""", (tag, tag, args.rows))
            cur.execute('ANALYZE papers')
        timings = []
        for _ in range(10):
            start = time.perf_counter()
            result = PaperRepository.hybrid_search('retrieval', Model(), embedding_model_name=tag)
            timings.append((time.perf_counter()-start)*1000)
            if not result['total_items']:
                raise RuntimeError('Benchmark corpus unexpectedly empty; do not run against a database being reset')
        with get_cursor() as cur:
            cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT id FROM papers WHERE embedding_model=%s ORDER BY embedding <=> %s::vector LIMIT 500", (tag, '[1,' + '0,'*766 + '1]'))
            plan = cur.fetchone()['QUERY PLAN']
        report = {'synthetic': True, 'rows': args.rows, 'runs': 10,
                  'median_ms': statistics.median(timings), 'max_ms': max(timings),
                  'candidate_count': result['total_items'], 'vector_plan': plan}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str)+'\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'vector_plan'}))
    finally:
        with get_cursor() as cur:
            cur.execute('DELETE FROM papers WHERE embedding_model=%s', (tag,))


if __name__ == '__main__':
    main()
