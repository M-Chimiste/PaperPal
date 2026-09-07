"""Record the running API's retrieval results for offline evaluation (no writes)."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
from theseus_insight.observability import provenance


def canonical(url):
    match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', url)
    return match.group(1) if match else url.rstrip('/')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://localhost:8000')
    parser.add_argument('--dataset', type=Path, default=Path('tests/evaluation/research_v1.json'))
    parser.add_argument('--output', type=Path, default=Path('data/evaluation/predictions.json'))
    args = parser.parse_args()
    headers = {'Content-Type': 'application/json'}
    token = os.getenv('APP_AUTH_TOKEN')
    if token:
        headers['Authorization'] = 'Basic ' + base64.b64encode(('theseus:' + token).encode()).decode()
    def request(path, body=None):
        req = Request(args.base_url.rstrip('/') + path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
        with urlopen(req, timeout=120) as response:
            return json.load(response)
    initial_provenance = request('/api/runtime/provenance')
    dataset = json.loads(args.dataset.read_text())
    ids = {canonical(p['url']): p['id'] for p in dataset['corpus']}
    cases = []
    for case in dataset['cases']:
        started = time.perf_counter()
        result = request('/api/papers/hybrid-search', {'query_text': case['query'], 'page_size': 20})
        cases.append({'id': case['id'], 'ranked_ids': [ids.get(canonical(p['url']), p['url']) for p in result['results']],
                      'latency_ms': (time.perf_counter()-started)*1000, 'cost_usd': None, 'citation_reviews': []})
    metadata = request('/api/runtime/provenance')
    if any(metadata.get(key) != initial_provenance.get(key) for key in ('config_hash', 'prompt_hash', 'code_hash')):
        raise RuntimeError('Deployment or configuration changed during evaluation; rerun under a fixed configuration')
    metadata['source'] = 'running-api'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'provenance': metadata, 'cases': cases}, indent=2)+'\n')
    print(f'Recorded {len(cases)} queries to {args.output}')


if __name__ == '__main__':
    main()
