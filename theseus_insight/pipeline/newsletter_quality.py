"""Local inference, source-anchored evidence, validated briefs, editorial selection."""
import hashlib
import copy
import json
import logging
import re
import time
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..prompt.newsletter_quality import SYSTEM, EXTRACT, WRITE, VERIFY, RANK, INTRO, INTRO_REVIEW


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Claim(StrictModel):
    category: Literal['problem', 'contribution', 'result', 'limitation']
    text: str = Field(min_length=12, max_length=1800)
    source_id: str
    quote: str = Field(min_length=20, max_length=2000)


class Extraction(StrictModel):
    claims: list[Claim] = Field(max_length=16)


class CitedText(StrictModel):
    text: str = Field(min_length=25, max_length=2400)
    evidence_ids: list[str]


class Brief(StrictModel):
    what_changed: CitedText
    evidence: CitedText
    why_it_matters: CitedText
    caveat: CitedText


class FieldCheck(StrictModel):
    field: Literal['what_changed', 'evidence', 'why_it_matters', 'caveat']
    supported: bool
    reason: str = Field(min_length=3)


class Review(StrictModel):
    checks: list[FieldCheck]


class EditorialScore(StrictModel):
    relevance: int = Field(ge=0, le=5)
    evidence_strength: int = Field(ge=0, le=5)
    novelty: int = Field(ge=0, le=5)
    usefulness: int = Field(ge=0, le=5)
    topic: str = Field(min_length=2, max_length=100)
    reason: str = Field(min_length=10, max_length=1000)


MISSING_LIMITATION = 'No explicit limitation was found in the extracted evidence.'


def require_local(client):
    if client.provider not in {'lmstudio', 'ollama', 'llamacpp'}:
        raise ValueError('Newsletter quality inference requires a local model provider')


def structured(client, instruction, payload, schema, validate=None):
    """Retry invalid content once; never publish raw/partially repaired output."""
    require_local(client)
    # Providers cache clients across roles. Use a shallow request-local copy so
    # role budgets never mutate shared clients or the user's saved settings.
    client = copy.copy(client)
    budget = {'Extraction': 4096, 'Brief': 1800, 'Review': 1200,
              'EditorialScore': 500, 'Intro': 1000, 'IntroReview': 800}.get(schema.__name__, 1800)
    if hasattr(client, 'max_new_tokens'):
        client.max_new_tokens = min(client.max_new_tokens or budget, budget)
    correction = ''
    failure = ''
    for _ in range(2):
        started = time.monotonic()
        response = client.invoke(
            messages=[{'role': 'user', 'content': instruction + '\n' + json.dumps(payload, ensure_ascii=False) + correction}],
            system_prompt=SYSTEM, schema=schema,
        )
        logging.getLogger(__name__).info('Newsletter %s inference completed in %.1fs', schema.__name__, time.monotonic() - started)
        try:
            result = schema.model_validate_json(response) if isinstance(response, str) else schema.model_validate(response)
            if validate:
                validate(result)
            return result
        except (ValueError, TypeError) as exc:
            failure = '; '.join('.'.join(map(str, e['loc'])) + ': ' + e['msg'] for e in exc.errors()[:4]) if isinstance(exc, ValidationError) else str(exc)
            logging.getLogger(__name__).warning('Newsletter %s validation retry: %s', schema.__name__, failure[:500])
            correction = '\nCorrect the previous validation failure and return a complete object: ' + failure[:1200]
    raise ValueError('Local model failed newsletter schema or evidence validation: ' + failure[:1000])


def normalize(text):
    return ' '.join(text.split())


def source_chunks(markdown, size=14000):
    """Bound each input, preserve exact text and offsets; never silently truncate."""
    if len(markdown.strip()) < 250:
        raise ValueError('Insufficient extracted paper text')
    if len(markdown) > size * 48:
        raise ValueError('Paper exceeds evidence extraction budget; review separately')
    chunks = []
    start = 0
    while start < len(markdown):
        end = min(start + size, len(markdown))
        if end < len(markdown):
            boundary = markdown.rfind('\n\n', start + size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append({'source_id': f'S{len(chunks)+1}', 'start': start, 'end': end, 'text': markdown[start:end]})
        start = end
    return chunks


def extract_evidence(client, markdown):
    chunks = source_chunks(markdown)
    claims = []
    seen = set()
    discarded_quotes = []
    for chunk in chunks:
        accepted = []
        rejected = []
        for attempt in range(2):
            instruction = EXTRACT
            if rejected:
                instruction += '\nCorrect or omit these non-verbatim claims: ' + json.dumps(rejected, ensure_ascii=False)
            result = structured(client, instruction, chunk, Extraction)
            rejected = []
            for claim in result.claims:
                if claim.source_id == chunk['source_id'] and normalize(claim.quote) in normalize(chunk['text']):
                    accepted.append(claim)
                else:
                    rejected.append(claim.model_dump())
            if not rejected:
                break
            logging.getLogger(__name__).warning('Newsletter %s: %d non-verbatim quotes on extraction attempt %d',
                                               chunk['source_id'], len(rejected), attempt + 1)
        # A failed quote cannot enter the evidence, but does not invalidate other
        # independently verified claims. Retain omissions for source-review audits.
        discarded_quotes.extend({'chunk_id': chunk['source_id'], **claim} for claim in rejected)
        for claim in accepted:
            identity = (claim.category, normalize(claim.quote))
            if identity not in seen:
                seen.add(identity)
                claims.append({'id': f'E{len(claims)+1}', **claim.model_dump()})
    if not {'problem', 'contribution'} <= {c['category'] for c in claims}:
        raise ValueError('Paper evidence validation failed: lacks a supported problem or contribution')
    if len(json.dumps(claims)) > 100000:
        raise ValueError('Extracted evidence exceeds the writing context budget; review separately')
    # Persist the original text for reproducible source review and fixed-input evaluation.
    return {'source_hash': hashlib.sha256(markdown.encode()).hexdigest(), 'sources': chunks,
            'claims': claims, 'discarded_quotes': discarded_quotes}


def validate_brief(brief, evidence, word_limit):
    valid_ids = {c['id'] for c in evidence['claims']}
    for name in Brief.model_fields:
        item = getattr(brief, name)
        if name == 'caveat' and item.text == MISSING_LIMITATION and not item.evidence_ids:
            if any(c['category'] == 'limitation' for c in evidence['claims']):
                raise ValueError('The evidence includes limitations; report one rather than the missing-evidence disclaimer')
            continue
        if not item.evidence_ids or not set(item.evidence_ids) <= valid_ids:
            raise ValueError(f'{name} requires valid supporting evidence IDs')
        if re.search(r'<[^>]+>|https?://|\[[^\]]*\]\(', item.text):
            raise ValueError('Draft fields must be plain text; links are rendered from metadata')
        if re.search(r'\b(game.changer|groundbreaking|incredibly exciting|breakthrough)\b', item.text, re.I):
            raise ValueError('Replace promotional language with concrete findings')
    words = sum(len(getattr(brief, name).text.split()) for name in Brief.model_fields)
    if not 70 <= words <= word_limit:
        raise ValueError(f'Brief must contain 70–{word_limit} words, got {words}')


def write_brief(writer, verifier, evidence, interests, featured=True, editorial_feedback=None):
    limit = 280 if featured else 160
    instruction = WRITE + f' Total word budget across fields: 70–{limit} words.'
    payload = {'research_interests': interests, 'evidence': evidence['claims']}
    if editorial_feedback:
        payload['editorial_feedback'] = editorial_feedback
    for attempt in range(2):
        brief = structured(writer, instruction, payload, Brief, lambda b: validate_brief(b, evidence, limit))
        def validate_review(review):
            fields = [check.field for check in review.checks]
            if len(fields) != 4 or set(fields) != set(Brief.model_fields):
                raise ValueError('Review must cover each draft field exactly once')
        review = structured(verifier, VERIFY, {'draft': brief.model_dump(), 'evidence': evidence['claims']}, Review, validate_review)
        if all(check.supported for check in review.checks):
            return brief
        payload['revision_instructions'] = [check.model_dump() for check in review.checks if not check.supported]
    reasons = '; '.join(f'{check.field}: {check.reason}' for check in review.checks if not check.supported)
    raise ValueError('Draft failed source support review after revision: ' + reasons)


def plain(text):
    """Escape Markdown syntax before HTML, so entity markers stay intact."""
    import html
    text = re.sub(r'([\\`*_{\[\]}()#!|])', r'\\\1', normalize(text))
    return html.escape(text, quote=False)


def _paper_anchor(paper_id):
    if not re.fullmatch(r'P[1-9]\d*', paper_id):
        raise ValueError('Invalid newsletter paper ID')
    return f'paper-{paper_id}'


def render_brief(title, url, brief, evidence, *, paper_id='P1'):
    anchor = _paper_anchor(paper_id)
    if urlsplit(url).scheme not in {'http', 'https'}:
        raise ValueError('Paper link must be HTTP(S)')
    safe_url = url.replace(' ', '%20').replace('(', '%28').replace(')', '%29').replace('\n', '')
    labels = {'what_changed': 'What Changed', 'why_it_matters': 'Why it matters',
              'evidence': 'Evidence', 'caveat': 'Caveat'}
    lines = [f'<a id="{anchor}" name="{anchor}"></a>', f'## {paper_id[1:]}. {plain(title)}',
             f'[Read paper](<{safe_url.replace(">", "%3E").replace("<", "%3C")}>)']
    valid_ids = {claim['id'] for claim in evidence['claims']}
    for field, label in labels.items():
        value = getattr(brief, field)
        if not set(value.evidence_ids) <= valid_ids:
            raise ValueError('Citation has no source note')
        if any(not re.fullmatch(r'E[1-9]\d*', ident) for ident in value.evidence_ids):
            raise ValueError('Invalid evidence ID')
        refs = ', '.join(f'[{paper_id}·{plain(ident)}](#source-{paper_id}-{ident})' for ident in value.evidence_ids)
        lines.append(f'**{label}:** {plain(value.text)}' + (f' [{refs}]' if refs else ''))
    return '\n\n'.join(lines)


def render_source_notes(papers):
    lines = ['## Source notes']
    for i, paper in enumerate(papers, 1):
        paper_id = f'P{i}'
        brief = Brief.model_validate(paper['brief'])
        used = {ident for field in Brief.model_fields for ident in getattr(brief, field).evidence_ids}
        lines.extend([f'### {i}. {plain(paper["title"])}', f'[Back to paper](#{_paper_anchor(paper_id)})'])
        for claim in paper['evidence']['claims']:
            if claim['id'] in used:
                if not re.fullmatch(r'E[1-9]\d*', claim['id']):
                    raise ValueError('Invalid evidence ID')
                quote_words = claim['quote'].split()
                excerpt = ' '.join(quote_words[:25]) + (' …' if len(quote_words) > 25 else '')
                lines.append(f'- <a id="source-{paper_id}-{claim["id"]}" name="source-{paper_id}-{claim["id"]}"></a>'
                             f' **{paper_id}·{claim["id"]}** ({plain(claim["source_id"])}): “{plain(excerpt)}”')
    return '\n\n'.join(lines)


def assemble_newsletter(intro, papers):
    """Render paper-scoped citations and a single source appendix in issue order."""
    sections = [render_brief(p['title'], p['url'], Brief.model_validate(p['brief']),
                p['evidence'], paper_id=f'P{i}') for i, p in enumerate(papers, 1)]
    return '\n\n---\n\n'.join([intro, *sections, render_source_notes(papers), '~Theseus Insight'])


def paper_key(row):
    # Treat arXiv versions and abs/pdf URLs as one paper for repeat coverage.
    url = str(row['pdf_url'])
    match = re.search(r'arxiv\.org/(?:abs|pdf)/(.+?)(?:\.pdf)?$', url)
    return re.sub(r'v\d+$', '', match.group(1)) if match else url


def editorial_candidates(rows, client, interests, previous=()):
    """Rank a bounded shortlist, then diversify deterministically by topic/title."""
    unique = {paper_key(row): row for row in reversed(rows)}
    candidates = []
    for key, row in unique.items():
        score = structured(client, RANK, {'title': row['title'], 'abstract': row['abstract'], 'interests': interests}, EditorialScore)
        if score.relevance < 2:
            continue
        base = 3*score.relevance + score.evidence_strength + score.novelty + score.usefulness
        candidates.append({**row, 'editorial': score.model_dump(), '_base': base - (8 if key in previous else 0)})
    selected = []
    topic_counts = {}
    while candidates:
        def priority(row):
            topic = normalize(row['editorial']['topic']).casefold()
            tokens = set(re.findall(r'\w+', row['title'].lower()))
            overlap = max((len(tokens & set(re.findall(r'\w+', s['title'].lower()))) / max(len(tokens | set(re.findall(r'\w+', s['title'].lower()))), 1) for s in selected), default=0)
            return (-row['_base'] + 3*topic_counts.get(topic, 0) + 5*overlap, paper_key(row))
        row = min(candidates, key=priority)
        candidates.remove(row)
        topic = normalize(row['editorial']['topic']).casefold()
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        selected.append(row)
    return selected


class SummaryParagraph(StrictModel):
    text: str = Field(min_length=40, max_length=1800)
    paper_ids: list[str] = Field(min_length=1)


class Intro(StrictModel):
    paragraphs: list[SummaryParagraph] = Field(min_length=1, max_length=3)


class IntroReview(StrictModel):
    supported: bool
    issues: list[str]


def build_intro(client, papers):
    inputs = [{'id': f'P{i+1}', 'title': p['title'], 'brief': p['brief']} for i, p in enumerate(papers)]
    def validate(intro):
        valid = {p['id'] for p in inputs}
        covered = set()
        for paragraph in intro.paragraphs:
            if not set(paragraph.paper_ids) <= valid or not 20 <= len(paragraph.text.split()) <= 120:
                raise ValueError('Summary paragraphs need valid paper IDs and 20–120 words')
            if re.search(r'https?://|<[^>]+>', paragraph.text):
                raise ValueError('Summary text must be plain prose; references belong in paper_ids')
            if not set(re.findall(r'\bP\d+\b', paragraph.text)) <= set(paragraph.paper_ids):
                raise ValueError('Inline paper references must match paragraph paper_ids')
            covered.update(paragraph.paper_ids)
        if covered != valid:
            raise ValueError('The preamble must summarize every included paper')
        if not 40 <= sum(len(p.text.split()) for p in intro.paragraphs) <= 220:
            raise ValueError('Issue summary must contain 40–220 words')
    intro = structured(client, INTRO, inputs, Intro, validate)
    review = structured(client, INTRO_REVIEW, {'papers': inputs, 'intro': intro.model_dump()}, IntroReview)
    if not review.supported or review.issues:
        raise ValueError('Introduction failed source-support review')
    # Some local models repeat reference markers in prose despite the schema.
    # Remove only citation markers; retain the validated paragraph-level references.
    return '\n\n'.join(plain(re.sub(r'\s*[\[(]P\d+(?:\s*[,;]\s*P\d+)*[\])]', '', p.text))
                       + ' (' + ', '.join(p.paper_ids) + ')' for p in intro.paragraphs)


def intro_excerpts(papers):
    """Faithful prose overview of all contributions if synthesis fails review."""
    return 'This issue presents ' + str(len(papers)) + ' research papers. ' + ' '.join(
        plain(p['brief']['what_changed']['text']) + f' (P{i+1})'
        for i, p in enumerate(papers))


def intro_or_excerpts(client, papers):
    """Use the same verified synthesis/fallback policy in production and evaluation."""
    try:
        return build_intro(client, papers), False
    except ValueError:
        return intro_excerpts(papers), True
