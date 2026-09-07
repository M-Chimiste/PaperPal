import asyncio
import json
from types import SimpleNamespace

import pytest

from theseus_insight.pipeline.newsletter_quality import (
    Claim, Brief, CitedText, Extraction, Review, EditorialScore, structured,
    extract_evidence, validate_brief, write_brief, editorial_candidates, paper_key,
    source_chunks, MISSING_LIMITATION,
)


class Client:
    provider = 'lmstudio'
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(next(self.responses))


@pytest.fixture
def evidence():
    text = 'The method reduces redundant search while preserving accuracy on the evaluated benchmark.'
    return {'source_hash': 'test', 'sources': [{'source_id': 'S1', 'text': text}],
            'claims': [{'id': 'E1', 'category': 'contribution', 'source_id': 'S1', 'text': text, 'quote': text}]}


@pytest.fixture
def brief():
    return Brief(
        what_changed=CitedText(text='The proposed method groups repeated search states and allocates further exploration to distinct alternatives instead of repeating the same approach.', evidence_ids=['E1']),
        evidence=CitedText(text='The reported evaluation finds reduced redundant search while preserving accuracy on the tested benchmark; this result describes the evaluated setting only.', evidence_ids=['E1']),
        why_it_matters=CitedText(text='For a reader investigating agent search, this could motivate an experiment comparing distinct explored states at a fixed inference budget in their own application.', evidence_ids=['E1']),
        caveat=CitedText(text=MISSING_LIMITATION, evidence_ids=[]))


def test_chunks_reconstruct_source_without_loss():
    text = ('A paragraph with measured results.\n\n' * 1000)
    chunks = source_chunks(text, size=1000)
    assert ''.join(c['text'] for c in chunks) == text
    assert all(len(c['text']) <= 1000 for c in chunks)


def test_extraction_rejects_fabricated_quotes():
    claim = {'category': 'contribution', 'text': 'An unsupported contribution was claimed.', 'quote': 'This sentence does not occur in the source.', 'source_id': 'S1'}
    client = Client([{'claims': [claim]}] * 2)
    with pytest.raises(ValueError, match='validation'):
        extract_evidence(client, 'Actual paper content about a different method. ' * 20)
    assert all(call['schema'] is Extraction for call in client.calls)


def test_schema_does_not_accept_raw_fallback():
    client = Client([{'draft': 'This document discusses UrbanGround'}] * 2)
    with pytest.raises(ValueError):
        structured(client, 'Write', {}, Brief)


def test_extraction_keeps_verified_claims_and_audits_discarded_quotes():
    source = 'Search repeats identical states and wastes compute. The method groups duplicate states before expanding them. ' * 4
    valid = [
        dict(category='problem', text='Repeated states consume unnecessary computation.',
             quote='Search repeats identical states and wastes compute.', source_id='S1'),
        dict(category='contribution', text='Duplicate states are grouped before expansion.',
             quote='The method groups duplicate states before expanding them.', source_id='S1'),
    ]
    invalid = dict(category='result', text='The method improves accuracy by ninety percent.',
                   quote='Accuracy improved by ninety percent in all tests.', source_id='S1')
    client = Client([{'claims': valid + [invalid]}] * 2)
    result = extract_evidence(client, source)
    assert len(result['claims']) == 2
    assert result['discarded_quotes'] == [{'chunk_id': 'S1', **invalid}]
    assert all(claim['quote'] in source for claim in result['claims'])
    assert 'Correct or omit' in client.calls[1]['messages'][0]['content']


def test_cloud_provider_is_rejected():
    client = Client([])
    client.provider = 'openai'
    with pytest.raises(ValueError, match='local'):
        structured(client, 'Write', {}, Brief)
    assert not client.calls


def test_brief_requires_real_evidence_and_complete_content(brief, evidence):
    validate_brief(brief, evidence, 280)
    brief.what_changed.evidence_ids = ['E999']
    with pytest.raises(ValueError, match='evidence IDs'):
        validate_brief(brief, evidence, 280)


def test_reviewer_must_cover_every_field(brief, evidence):
    writer = Client([brief.model_dump()])
    verifier = Client([{'checks': []}] * 2)
    with pytest.raises(ValueError):
        write_brief(writer, verifier, evidence, 'agent search')


def test_failed_support_review_revises_then_rejects(brief, evidence):
    checks = [{'field': name, 'supported': name != 'evidence', 'reason': 'Missing baseline qualification'} for name in Brief.model_fields]
    writer = Client([brief.model_dump()] * 2)
    verifier = Client([{'checks': checks}] * 2)
    with pytest.raises(ValueError, match='support review'):
        write_brief(writer, verifier, evidence, 'agent search')
    assert 'revision_instructions' in writer.calls[1]['messages'][0]['content']


def test_editorial_feedback_is_passed_to_local_writer(brief, evidence):
    writer = Client([brief.model_dump()])
    checks = [{'field': name, 'supported': True, 'reason': 'Supported by the cited evidence'} for name in Brief.model_fields]
    verifier = Client([{'checks': checks}])
    write_brief(writer, verifier, evidence, 'agent search', editorial_feedback='Limit results to the tested backbone.')
    assert 'Limit results to the tested backbone.' in writer.calls[0]['messages'][0]['content']


def test_editorial_deduplication_repeat_penalty_and_relevance():
    rows = [{'title': name, 'abstract': name, 'pdf_url': url} for name, url in [
        ('Prior paper', 'https://arxiv.org/pdf/2601.00001v2'),
        ('Prior paper again', 'https://arxiv.org/abs/2601.00001v1'),
        ('Fresh paper', 'https://arxiv.org/pdf/2601.00002'),
        ('Unrelated', 'https://arxiv.org/pdf/2601.00003')]]
    # Order is reversed by stable deduplication; unrelated, fresh, prior.
    def score(relevance):
        return dict(relevance=relevance, evidence_strength=3, novelty=3, usefulness=3, topic='search', reason='Direct relevance to search methods')
    client = Client([score(0), score(4), score(4)])
    selected = editorial_candidates(rows, client, 'search', previous=['2601.00001'])
    assert [r['title'] for r in selected] == ['Fresh paper', 'Prior paper']
    assert paper_key(rows[0]) == paper_key(rows[1])


@pytest.mark.asyncio
async def test_download_speed_cannot_change_selection(tmp_path, monkeypatch, evidence, brief):
    import time
    import pandas as pd
    from theseus_insight.pipeline.stages import newsletter_sections as stage
    rows = [{'title': name, 'pdf_url': 'https://example.org/' + name, 'abstract': name, 'editorial': {}} for name in ['best', 'backup']]
    monkeypatch.setattr(stage, 'editorial_candidates', lambda *args: rows)
    monkeypatch.setattr(stage, 'extract_evidence', lambda *args: evidence)
    monkeypatch.setattr(stage, 'write_brief', lambda *args: brief)
    checkpoints = {}
    async def load(name): return checkpoints.get(name)
    async def save(name, value): checkpoints[name] = value
    def download(url):
        name = url.rsplit('/', 1)[-1]
        if name == 'best': time.sleep(.08)
        path = tmp_path / (name + '.pdf')
        path.write_text('fake PDF')
        return str(path)
    ti = SimpleNamespace(top_n=1, db_saving=False, task_id=None, research_interests='search',
                         content_extraction_inference=Client([]), newsletter_sections_inference=Client([]),
                         pdf_download_max_workers=2, _load_checkpoint_async=load, _save_checkpoint_async=save,
                         _download_pdf_to_temp_file=download, _parse_downloaded_pdf_to_markdown=lambda *a: 'text')
    result = await stage.run(ti, pd.DataFrame(rows), None, None)
    assert result['papers'][0]['title'] == 'best'
    assert not list(tmp_path.glob('*.pdf'))


@pytest.mark.asyncio
async def test_intro_receives_papers_not_characters(monkeypatch, brief, evidence):
    from theseus_insight.pipeline.stages import newsletter_content as stage
    papers = [{'key': 'paper', 'title': 'A paper', 'url': 'https://example.org/paper', 'brief': brief.model_dump(), 'evidence': evidence}]
    def intro(client, inputs):
        assert inputs == papers
        return 'Dear Reader,\n\nA grounded introduction.', False
    monkeypatch.setattr(stage, 'intro_or_excerpts', intro)
    async def load(name): return None
    async def save(name, value): pass
    ti = SimpleNamespace(db_saving=False, newsletter_intro_inference=Client([]), _load_checkpoint_async=load, _save_checkpoint_async=save)
    content, _ = await stage.run(ti, {'papers': papers, 'sections': ['A complete section']}, None, None)
    assert 'Paper P1' not in content
    assert '## 1. A paper' in content
    assert '\n\n---\n\n' in content
    assert content.endswith('~Theseus Insight')


def test_local_preview_keeps_styled_html_and_markdown(tmp_path):
    from scripts.newsletter_eval import save_preview
    content = '# Theseus Insight\n\n---\n\n## A paper\n\n**Evidence:** Supported result.'
    output = tmp_path / 'draft.json'
    save_preview(output, content)
    assert output.with_suffix('.md').read_text() == content
    html = output.with_suffix('.html').read_text()
    assert '<hr' in html
    assert '<strong>Evidence:</strong>' in html
    assert 'title-block' in html
    assert 'border-bottom: 2px solid #3498db' in html
    with pytest.raises(ValueError, match='overwrite'):
        save_preview(output, 'replacement')


def test_source_appendix_links_are_scoped_by_paper(brief, evidence):
    from html.parser import HTMLParser
    from theseus_insight.pipeline.newsletter_quality import assemble_newsletter
    from theseus_insight.communication.rendering import render_newsletter_html
    papers = [{'title': title, 'url': f'https://example.org/{i}',
               'brief': brief.model_dump(), 'evidence': evidence}
              for i, title in enumerate(['First paper', 'Second paper'])]
    content = assemble_newsletter('Introduction', papers)
    body, notes = content.split('## Source notes', 1)
    assert body.index('**What Changed:**') < body.index('**Why it matters:**') < body.index('**Evidence:**') < body.index('**Caveat:**')
    assert 'source-P1-E1' in notes and 'source-P2-E1' in notes
    assert '### 1. First paper' in notes and '### 2. Second paper' in notes
    assert evidence['claims'][0]['quote'] not in body

    class Anchors(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids, self.links, self.names = [], [], []
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if 'id' in attrs: self.ids.append(attrs['id'])
            if tag == 'a' and 'name' in attrs: self.names.append(attrs['name'])
            if tag == 'a': self.links.append(attrs.get('href', ''))
    parsed = Anchors()
    parsed.feed(render_newsletter_html(content))
    assert len(parsed.ids) == len(set(parsed.ids))
    assert {'#source-P1-E1', '#source-P2-E1', '#paper-P1', '#paper-P2'} <= set(parsed.links)
    assert all(href[1:] in parsed.ids for href in parsed.links if href.startswith('#'))
    assert all(href[1:] in parsed.names for href in parsed.links if href.startswith('#'))
    assert 'https://example.org/0' in parsed.links


def test_plain_text_entities_render_without_corruption_or_html_injection():
    import html
    import re
    from theseus_insight.pipeline.newsletter_quality import plain
    from theseus_insight.communication.rendering import render_newsletter_html
    source = '''The 'seed lottery' & "memory" <script>alert(1)</script> # result'''
    rendered = render_newsletter_html(plain(source))
    paragraph = re.search(r'<p>(.*?)</p>', rendered, re.S).group(1)
    assert html.unescape(paragraph) == source
    assert '&#x27;' not in paragraph and '&amp;#x27;' not in paragraph
    assert '<script>' not in rendered


def test_preamble_covers_all_papers_in_prose(brief):
    from theseus_insight.pipeline.newsletter_quality import build_intro, intro_excerpts
    papers = [{'title': f'Paper {i}', 'brief': brief.model_dump()} for i in range(5)]
    paragraph = 'The work examines how structured search can reduce repeated computation while preserving measured accuracy in the evaluated setting, with potential applications requiring separate validation. '
    incomplete = {'paragraphs': [{'text': paragraph * 2, 'paper_ids': ['P1', 'P2']}]}
    with pytest.raises(ValueError, match='every included paper'):
        build_intro(Client([incomplete] * 2), papers)
    complete = {'paragraphs': [{'text': paragraph * 2, 'paper_ids': [f'P{i}' for i in range(1, 6)]}]}
    output = build_intro(Client([complete, {'supported': True, 'issues': []}]), papers)
    assert not output.startswith('Dear Reader') and '\n- ' not in output
    assert 'P5' in output
    fallback = intro_excerpts(papers)
    assert all(f'(P{i})' in fallback for i in range(1, 6))
    assert '\n- ' not in fallback


def test_missing_limitation_disclaimer_cannot_hide_known_limitations(brief, evidence):
    evidence['claims'].append({'id': 'E2', 'category': 'limitation'})
    with pytest.raises(ValueError, match='includes limitations'):
        validate_brief(brief, evidence, 280)


def test_evaluation_rejects_mismatched_inputs_and_preserves_unknown_ratings(tmp_path):
    from scripts.newsletter_eval import blind, report
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(json.dumps({'fixture_hash': 'a'}))
    b.write_text(json.dumps({'fixture_hash': 'b'}))
    with pytest.raises(ValueError, match='same frozen input'):
        blind(SimpleNamespace(variants=[a, b], output=tmp_path / 'blind'))
    ratings = tmp_path / 'ratings.json'
    ratings.write_text(json.dumps({'Variant-1': {'factual_support': None, 'readability': None}}))
    output = tmp_path / 'report.json'
    report(SimpleNamespace(ratings=ratings, output=output))
    data = json.loads(output.read_text())['Variant-1']
    assert data['coverage'] == 0
    assert data['mean_of_rated_dimensions'] is None


def test_role_budget_does_not_mutate_cached_client():
    class BudgetClient:
        provider = 'lmstudio'
        max_new_tokens = 16000
        def invoke(self, **kwargs):
            assert self.max_new_tokens == 500
            return json.dumps(dict(relevance=4, evidence_strength=3, novelty=3, usefulness=3,
                                   topic='Search methods', reason='Useful direct connection to search'))
    client = BudgetClient()
    structured(client, 'Score', {}, EditorialScore)
    assert client.max_new_tokens == 16000
