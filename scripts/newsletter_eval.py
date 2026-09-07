"""Freeze evidence, generate local-only draft variants, and prepare blind human review.

No email sending or database mutation. Generated variants use identical source papers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

METRICS = ['factual_support', 'specificity', 'relevance', 'readability', 'non_repetition']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError(f'Refusing to overwrite {target}')
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def snapshot(args):
    import os
    import psycopg
    from dotenv import load_dotenv
    load_dotenv()
    with psycopg.connect(os.environ['DATABASE_URL'], options='-c default_transaction_read_only=on') as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT artifact FROM newsletter_editions WHERE newsletter_id=%s', (args.newsletter_id,))
            row = cur.fetchone()
    if not row:
        raise ValueError('This issue has no evidence artifact. Use an issue generated with the quality pipeline.')
    # Research interests are explicit so the frozen test is reproducible after profile edits.
    fixture = {'version': 1, 'research_interests': Path(args.interests).read_text(), 'papers': row[0]['papers']}
    save(args.output, fixture)


def generate(args):
    from dotenv import load_dotenv
    load_dotenv()
    from theseus_insight.pipeline.model_loading import load_inference_model
    from theseus_insight.pipeline.newsletter_quality import extract_evidence, write_brief, assemble_newsletter, intro_or_excerpts
    from theseus_insight.observability import provenance
    fixture = read(args.fixture)
    config = read(args.config)
    clients = {}
    for role in ['content_extraction_model', 'newsletter_sections_model', 'newsletter_intro_model']:
        cfg = dict(config[role])
        if cfg['model_type'] not in {'lmstudio', 'ollama'}:
            raise ValueError('Evaluation accepts only local model configurations')
        if args.temperature is not None:
            cfg['temperature'] = args.temperature
        config[role] = cfg
    for role in ['content_extraction_model', 'newsletter_sections_model', 'newsletter_intro_model']:
        cfg = config[role]
        clients[role] = load_inference_model(cfg['model_type'], cfg['model_name'], cfg['max_new_tokens'], cfg['temperature'], cfg.get('num_ctx'), cfg.get('host'))
    started = time.monotonic()
    papers = []
    for i, paper in enumerate(fixture['papers']):
        evidence = paper['evidence']
        if not args.reuse_evidence:
            source = ''.join(chunk['text'] for chunk in evidence['sources'])
            evidence = extract_evidence(clients['content_extraction_model'], source)
        brief = write_brief(clients['newsletter_sections_model'], clients['content_extraction_model'], evidence,
                            fixture['research_interests'], i < 2)
        papers.append({**paper, 'brief': brief.model_dump(), 'evidence': evidence})
    intro, intro_fallback = intro_or_excerpts(clients['newsletter_intro_model'], papers)
    content = assemble_newsletter(intro, papers)
    save(args.output, {'fixture_hash': digest(fixture), 'provenance': provenance(config),
                       'models': {role: {key: config[role].get(key) for key in ['model_name', 'model_type', 'temperature']} for role in clients},
                       'reuse_evidence': args.reuse_evidence, 'intro_fallback': intro_fallback, 'latency_seconds': time.monotonic()-started,
                       'content': content, 'papers': papers, 'human_quality': None})
    save_preview(args.output, content)



def save_preview(output, content):
    from theseus_insight.communication.rendering import render_newsletter_html
    stem = Path(output)
    targets = {stem.with_suffix('.md'): content, stem.with_suffix('.html'): render_newsletter_html(content)}
    if any(path.exists() for path in targets):
        raise ValueError('Refusing to overwrite an existing newsletter preview')
    for path, value in targets.items():
        path.write_text(value)


def blind(args):
    variants = [(str(path), read(path)) for path in args.variants]
    if len({v['fixture_hash'] for _, v in variants}) != 1:
        raise ValueError('Blind comparisons must use the same frozen input')
    random.SystemRandom().shuffle(variants)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    key, ratings = {}, {}
    for i, (path, variant) in enumerate(variants):
        label = f'Variant-{i+1}'
        save_preview(directory / (label + '.json'), variant['content'])
        key[label] = {'path': path, 'latency_seconds': variant['latency_seconds']}
        ratings[label] = {metric: None for metric in METRICS}
    save(directory / 'answer-key.json', key)
    save(directory / 'ratings.json', ratings)
    (directory / 'README.md').write_text('Read the variants before opening answer-key.json. Rate each dimension 1–5 in ratings.json.\nCheck factual support against the frozen source evidence, not model confidence.\nMissing ratings remain unknown. Compare latency separately from quality.\n')


def report(args):
    ratings = read(args.ratings)
    output = {}
    for label, metrics in ratings.items():
        if set(metrics) - set(METRICS):
            raise ValueError('Unknown human-rating dimension')
        metrics = {name: metrics.get(name) for name in METRICS}
        values = [value for value in metrics.values() if value is not None]
        if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not 1 <= value <= 5 for value in values):
            raise ValueError('Human ratings must be numbers from 1 to 5, or null')
        output[label] = {'ratings': metrics, 'coverage': len(values) / max(len(metrics), 1),
                         'mean_of_rated_dimensions': sum(values) / len(values) if values else None}
    save(args.output, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('snapshot')
    freeze.add_argument('--newsletter-id', type=int, required=True)
    freeze.add_argument('--interests', required=True)
    freeze.add_argument('--output', required=True)
    variant = commands.add_parser('generate')
    variant.add_argument('--fixture', required=True)
    variant.add_argument('--config', default='config/orchestration.json')
    variant.add_argument('--temperature', type=float)
    variant.add_argument('--reuse-evidence', action='store_true')
    variant.add_argument('--output', required=True)
    comparison = commands.add_parser('blind')
    comparison.add_argument('variants', nargs='+')
    comparison.add_argument('--output', required=True)
    summary = commands.add_parser('report')
    summary.add_argument('--ratings', required=True)
    summary.add_argument('--output', required=True)
    args = parser.parse_args()
    {'snapshot': snapshot, 'generate': generate, 'blind': blind, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
