"""Evaluate recorded retrieval runs; --smoke checks wiring with a lexical baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from theseus_insight.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('tests/evaluation/research_v1.json'))
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('data/evaluation/latest.json'))
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--min-recall', type=float)
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text())
    if args.smoke:
        cases = []
        for case in data['cases']:
            started = time.perf_counter()
            words = set(re.findall(r'\w+', case['query'].lower()))
            ranked = sorted(data['corpus'], key=lambda paper: (-len(words & set(re.findall(r'\w+', (paper['title']+' '+paper['summary']).lower()))), paper['id']))
            cases.append({'id': case['id'], 'ranked_ids': [p['id'] for p in ranked], 'latency_ms': (time.perf_counter()-started)*1000, 'cost_usd': 0})
        predictions = {'cases': cases, 'provenance': {'model': 'lexical-smoke-only', 'production_quality_claim': False}}
    elif args.predictions:
        predictions = json.loads(args.predictions.read_text())
        required = {'model', 'prompt_hash', 'config_hash', 'code_revision'}
        if not required <= predictions.get('provenance', {}).keys():
            parser.error('Recorded predictions require model, prompt_hash, config_hash and code_revision provenance')
    else:
        parser.error('Supply --predictions or --smoke')
    report = evaluate(data, predictions)
    report['dataset_sha256'] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        if baseline.get('dataset_sha256') != report['dataset_sha256'] or baseline['k'] != report['k']:
            parser.error('Baseline must use the same dataset and k')
        report['delta'] = {key: value-baseline['summary'][key] for key, value in report['summary'].items()
                           if value is not None and baseline['summary'].get(key) is not None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2))
    if args.min_recall is not None and report['summary']['recall_at_k'] < args.min_recall:
        raise SystemExit('Recall gate failed')


if __name__ == '__main__':
    main()
