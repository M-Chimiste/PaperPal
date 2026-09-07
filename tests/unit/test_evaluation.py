import pytest
from theseus_insight.evaluation import evaluate


def test_metrics_penalize_wrong_results_and_unknown_citations():
    data = {'version': 1, 'cases': [{'id': 'q', 'relevant_ids': ['a', 'b']}]}
    result = evaluate(data, {'cases': [{'id': 'q', 'ranked_ids': ['wrong', 'a', 'a']}]}, k=2)
    assert result['summary']['recall_at_k'] == .5
    assert result['summary']['precision_at_k'] == .5
    assert 0 < result['summary']['ndcg_at_k'] < 1
    assert result['summary']['citation_support'] is None
    assert result['summary']['cost_usd'] is None
    assert result['summary']['citation_support_coverage'] == 0


def test_incomplete_predictions_are_not_silently_ignored():
    with pytest.raises(ValueError, match='exactly once'):
        evaluate({'version': 1, 'cases': [{'id': 'q', 'relevant_ids': ['a']}]}, {'cases': []})
