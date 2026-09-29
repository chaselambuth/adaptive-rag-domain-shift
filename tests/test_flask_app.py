import threading

import numpy as np
import pandas as pd
import pytest

from app.rag_service import LunaRAGService, display_text, display_title
from app.web import create_app
from adaptive_rag.luna_v2.policy import ConstantProbability


# A small frozen fixture exercises real replay/routing behavior without external data.
@pytest.fixture
def service(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    value = LunaRAGService.__new__(LunaRAGService)
    value.domains = ['techqa', 'natural_questions']
    value.docs = pd.DataFrame([dict(document_id='d1', domain='techqa', title='Guide', text='Restart the server.', source='fixture')])
    value.doc_lookup = value.docs.set_index('document_id')
    value.cohort = pd.DataFrame([dict(example_id='q1', domain='techqa', question='How do I restart?')])
    value.questions = {('techqa', 'how do i restart?'): 'q1'}
    value.ranked = pd.DataFrame([dict(example_id='q1', retrieval_method=method, document_ids=np.array(['d1']), scores=np.array([0.03]), latency_ms=12.) for method in ['hybrid_rrf', 'hybrid_rrf_reranked', 'bm25']]).set_index(['example_id', 'retrieval_method'])
    value.results = pd.DataFrame([dict(example_id='q1', condition=method, status='completed', generated_answer='Restart the server.', estimated_cost_usd=0.0002) for method in ['direct', 'hybrid_rrf', 'hybrid_rrf_reranked']]).set_index(['example_id', 'condition'])
    class QueryTransformer:
        def transform(self, frame):
            return frame.assign(question_token_length=5)
    value.policy = dict(transformer=QueryTransformer(), thresholds=[0., 1.01, 0.], stages=[
        dict(model=ConstantProbability(.6), columns=['question_token_length']),
        dict(model=ConstantProbability(.8), columns=['hybrid_top1']),
        dict(model=ConstantProbability(.9), columns=['hybrid_top1'])])
    value.cfg = dict(generation=dict(model='gpt-5.6-luna', context_top_k=5), retrieval=dict(candidate_depth=50, top_k=10))
    value.metrics = dict(direct_quality=.029, hybrid_quality=.143, adaptive_quality=.143, adaptive_abstain=.148)
    value.research_findings = dict(default_runtime_label='Hybrid RRF', generator='gpt-5.6-luna', corpus='Fixture', policy_version='luna-v2', points=[])
    value.examples = [dict(question='How do I restart?', label='Saved example', domain='techqa')]
    value.lock = threading.RLock()
    value.sparse, value.dense, value.dense_failed = {}, {}, set()
    value.live_answers = {}
    return value


def test_flask_pages_and_saved_replay(service):
    client = create_app(service).test_client()
    assert client.get('/').status_code == 200
    assert b'Adaptive RAG Explorer' in client.get('/').data
    assert client.get('/health').json['policy_version'] == 'luna-v2'
    answer = client.post('/ask', json=dict(question='How do I restart?', generate=True, use_saved=True)).json
    assert answer['generation']['status'] == 'saved_completed'
    assert answer['generation']['answer'] == 'Restart the server.'
    assert answer['runtime']['generation_ms'] is None
    assert answer['runtime']['total_ms'] is None
    assert answer['runtime']['replay'] is True
    assert answer['documents'][0]['document_id'] == 'd1'


def test_natural_questions_markup_is_readable_without_altering_artifacts():
    raw = "Ohm's law &amp; resistance <H1>Ohm's law</H1> <Table> <Tr>Voltage = current × resistance</Tr>"
    clean = display_text(raw)
    assert clean == "Ohm's law & resistance Ohm's law Voltage = current × resistance"
    assert display_title('', 'https://en.wikipedia.org/w/index.php?title=Ohm%27s_law&amp;oldid=1', clean) == "Ohm's law"


def test_sequential_policy_and_compare(service):
    client = create_app(service).test_client()
    result = client.post('/ask', json=dict(question='How do I restart?', mode='adaptive', generate=True, use_saved=True)).json
    assert result['final_action'] == 'RETRIEVE'
    assert len(result['policy_stages']) == 2  # Stage three is never evaluated on this route.
    comparison = client.post('/compare', json=dict(question='How do I restart?', generate=True, use_saved=True)).json
    assert len(comparison['rows']) == 4
    assert all(r['generation_status'] == 'saved_completed' for r in comparison['rows'])


def test_failed_saved_answer_is_unavailable(service):
    service.results.loc[('q1', 'hybrid_rrf'), 'status'] = 'failed'
    result = service.answer('How do I restart?', generate=True, use_saved=True)
    assert result['generation']['status'] == 'saved_failed'
    assert result['generation']['answer'] == ''


def test_new_bm25_query_without_api_key(service):
    result = service.answer('How to restart the server?', mode='bm25', generate=True)
    assert result['retrieval_method'] == 'bm25'
    assert result['runtime']['replay'] is False
    assert result['generation']['status'] == 'not_configured'
    assert any('time includes setup' in warning for warning in result['warnings'])


def test_live_is_default_even_for_an_experiment_question(service):
    result = service.answer('How do I restart?', mode='bm25', generate=True)
    assert result['example_id'] is None
    assert result['runtime']['replay'] is False
    assert result['generation']['status'] == 'not_configured'


def test_replay_requires_an_exact_saved_question(service):
    client = create_app(service).test_client()
    response = client.post('/ask', json=dict(question='Different question', use_saved=True))
    assert response.status_code == 400
    assert 'Disable saved replay' in response.json['error']


def test_live_fallback_does_not_use_bm25_as_hybrid_policy_signals(service):
    service.dense_failed.add('techqa')
    result = service.answer('How to restart the server?', mode='adaptive', generate=False)
    assert result['retrieval_method'] == 'bm25'
    assert len(result['policy_stages']) == 1
    assert any('frozen policy was not applied' in warning for warning in result['warnings'])


@pytest.mark.parametrize('status,answer,expected', [('completed', 'Restart.', 'live_completed'),
    ('incomplete', 'Partial', 'live_unavailable')])
def test_live_calls_cache_and_hide_incomplete_answers(service, monkeypatch, status, answer, expected):
    import openai
    from types import SimpleNamespace
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-test-key')
    service.cfg['generation'].update(reasoning_effort='none', max_output_tokens=256, context_chars_per_document=1200)
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status=status, output_text=answer, model='gpt-5.6-luna', model_dump=lambda: {'output': []})
    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(create=create)))
    first = service.answer('A new question', mode='direct', generate=True)
    second = service.answer('A new question', mode='direct', generate=True)
    assert first['generation']['status'] == expected
    assert len(calls) == 1
    assert first['generation']['answer'] == (answer if status == 'completed' else '')
    assert second['generation']['status'] == ('live_cached' if status == 'completed' else expected)


@pytest.mark.parametrize('payload', [None, [], dict(question=''), dict(question=4),
    dict(question='hello', top_k='bad'), dict(question='hello', top_k=1.5),
    dict(question='hello', top_k=True), dict(question='hello', generate='bad'),
    dict(question='hello', domain='other'), dict(question='hello', mode='unknown')])
def test_invalid_input_returns_400(service, payload):
    client = create_app(service).test_client()
    for endpoint in ['/ask', '/compare']:
        response = client.post(endpoint, json=payload)
        assert response.status_code == 400
        assert response.json['error']
