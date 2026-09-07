"""Offline quality metrics. Citation support is explicitly human-reviewed."""
import math
import statistics


def evaluate(dataset, predictions, k=5):
    if k < 1 or not dataset["cases"]:
        raise ValueError("Nonempty dataset and positive k are required")
    expected = {case['id']: case for case in dataset['cases']}
    supplied = {case['id']: case for case in predictions['cases']}
    if len(supplied) != len(predictions['cases']) or set(supplied) != set(expected):
        raise ValueError('Predictions must contain each benchmark case exactly once')
    rows = []
    for key, case in expected.items():
        prediction = supplied[key]
        ranked = list(dict.fromkeys(prediction['ranked_ids']))[:k]
        relevant = set(case['relevant_ids'])
        if not relevant:
            raise ValueError('Each benchmark query requires relevance judgments')
        hits = [int(item in relevant) for item in ranked]
        dcg = sum(hit / math.log2(i+2) for i, hit in enumerate(hits))
        ideal = sum(1 / math.log2(i+2) for i in range(min(k, len(relevant))))
        citation_reviews = prediction.get('citation_reviews', [])
        if any(review.get('supported') not in (True, False) for review in citation_reviews):
            raise ValueError('Citation reviews require explicit supported true/false labels')
        rows.append({'id': key, 'recall_at_k': sum(hits)/len(relevant),
                     'precision_at_k': sum(hits)/k, 'ndcg_at_k': dcg/ideal,
                     'citation_support': statistics.mean(review['supported'] for review in citation_reviews) if citation_reviews else None,
                     'latency_ms': prediction.get('latency_ms'), 'cost_usd': prediction.get('cost_usd')})
    summary = {}
    for metric in ('recall_at_k', 'precision_at_k', 'ndcg_at_k', 'citation_support', 'latency_ms', 'cost_usd'):
        values = [row[metric] for row in rows if row[metric] is not None]
        summary[metric] = statistics.mean(values) if values else None
        summary[metric + '_coverage'] = len(values)/len(rows)
    latencies = sorted(row['latency_ms'] for row in rows if row['latency_ms'] is not None)
    summary['latency_p95_ms'] = latencies[max(0, math.ceil(.95*len(latencies))-1)] if latencies else None
    return {'benchmark_version': dataset['version'], 'k': k, 'summary': summary,
            'provenance': predictions.get('provenance', {}), 'cases': rows}
