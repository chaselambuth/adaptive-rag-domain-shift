import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from adaptive_rag.luna_v2.common import ABSTAIN, atomic_json, file_hash, save_table
from adaptive_rag.luna_v2.dataset import annotation_labels, prepare_data, split_cohort
from adaptive_rag.luna_v2.evaluation import (evaluate_results, paired_complete_cases,
    replay_actions, require_complete)
from adaptive_rag.luna_v2.generation import (batch_step, generation_status,
    parse_batch_result, prepare_requests, read_json, result_table, run_batches)
from adaptive_rag.luna_v2.policy import SourceQueryTransformer
from adaptive_rag.luna_v2.policy import train_policy
from adaptive_rag.luna_v2.report import final_evaluation


def test_luna_notebooks_follow_report_format():
    root = Path(__file__).resolve().parents[1]
    notebooks = sorted((root / 'experiment_notebooks/luna_v2').glob('0[1-5]_*.ipynb'))
    assert len(notebooks) == 5
    for number, path in enumerate(notebooks, 1):
        notebook = nbformat.read(path, as_version=4)
        assert notebook.cells[0].source.startswith(f'# **Notebook {number:02d}:')
        assert '**Description:**' in notebook.cells[0].source
        assert '**Main Questions This Notebook Should Answer:**' in notebook.cells[0].source
        markdown = [cell.source for cell in notebook.cells if cell.cell_type == 'markdown']
        assert any('# **Part 1:' in source for source in markdown)
        assert any('### **Environment Setup**' in source for source in markdown)
        assert any('### **Figure Export Helper**' in source for source in markdown)
        assert any('# **Part ' in source and 'Conclusion**' in source for source in markdown)
        assert set(notebook.metadata) == {'kernelspec', 'language_info'}
        part_starts = [i for i, cell in enumerate(notebook.cells)
                       if cell.cell_type == 'markdown' and '# **Part ' in cell.source]
        for position, start in enumerate(part_starts):
            end = part_starts[position + 1] if position + 1 < len(part_starts) else len(notebook.cells)
            if 'Conclusion**' not in notebook.cells[start].source:
                assert any(cell.cell_type == 'code' for cell in notebook.cells[start + 1:end])


def config(tmp_path):
    processed = tmp_path / 'data' / 'luna_v2'
    artifacts = tmp_path / 'artifacts' / 'luna_v2'
    processed.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    return {'_root': tmp_path, 'version': 'test-luna-v2', 'seed': 7,
        'paths': {'processed': processed, 'artifacts': artifacts},
        'generation': {'model': 'gpt-5.6-luna', 'reasoning_effort': 'none',
            'max_output_tokens': 64, 'context_top_k': 1, 'context_chars_per_document': 100,
            'prompt_version': 'test', 'min_f1_for_correct': .5,
            'batch_input_per_million': .1, 'batch_output_per_million': .6,
            'reserve_headroom': 1.25, 'batch_max_input_tokens': 100_000,
            'batch_max_requests': 10, 'budget_usd': 1.},
        'policy': {'wrong_answer_penalty': .25, 'cost_weight': 100.}}


def cohort_frame():
    return pd.DataFrame([
        {'example_id': 'nq1', 'domain': 'natural_questions', 'question': 'Where is Alpha?',
         'answer': 'Paris', 'answer_aliases': ['Paris'], 'answerable': True,
         'gold_document_ids': ['d1'], 'source_split': 'train', 'metadata': '{}',
         'annotation_category': 'short_or_yes_no', 'generation_eligible': True,
         'split': 'nq_test', 'split_group': 'g1', 'cross_domain_duplicate': False},
        {'example_id': 'tq1', 'domain': 'techqa', 'question': 'Unsupported setting?',
         'answer': '', 'answer_aliases': [], 'answerable': False,
         'gold_document_ids': [], 'source_split': 'dev', 'metadata': '{}',
         'annotation_category': 'benchmark_unanswerable', 'generation_eligible': True,
         'split': 'techqa_test', 'split_group': 'tq1', 'cross_domain_duplicate': False},
    ])


def rankings_frame():
    rows = []
    for example_id, domain, doc_id in [('nq1', 'natural_questions', 'd1'), ('tq1', 'techqa', 'd2')]:
        for method, latency in [('hybrid_rrf', 4.), ('hybrid_rrf_reranked', 9.)]:
            rows.append({'example_id': example_id, 'domain': domain, 'retrieval_method': method,
                         'document_ids': [doc_id], 'scores': [1.], 'latency_ms': latency,
                         'metadata': '{}'})
    return pd.DataFrame(rows)


def test_nq_annotations_join_multispan_and_do_not_invent_gold():
    record = {'document_tokens': [{'token': x} for x in 'a b c d e'.split()], 'annotations': [
        {'yes_no_answer': 'NONE', 'short_answers': [
            {'start_token': 1, 'end_token': 2}, {'start_token': 3, 'end_token': 5}],
         'long_answer': {'start_token': 0, 'end_token': 5}}]}
    assert annotation_labels(record) == (['b d e'], True, 'short_or_yes_no')
    assert annotation_labels({'annotations': [{'long_answer': {'start_token': 2, 'end_token': 8}}]}) == (
        [], True, 'long_only')
    assert annotation_labels({'document_text': 'page text', 'annotations': [
        {'long_answer': {'start_token': -1, 'end_token': -1}, 'short_answers': []}]}) == (
        [], False, 'no_positive_annotation')


def test_group_split_keeps_duplicate_questions_and_pages_together():
    rows = []
    for i in range(20):
        rows.append({'example_id': f'n{i}', 'domain': 'natural_questions',
            'question': f'Question {i // 2}', 'answerable': True,
            'gold_document_ids': [f'doc{i // 2}'], 'generation_eligible': True,
            'metadata': {}, 'answer': 'x', 'answer_aliases': ['x'], 'source_split': 'train'})
    rows.append({'example_id': 't', 'domain': 'techqa', 'question': 'Question 0',
        'answerable': True, 'gold_document_ids': ['td'], 'generation_eligible': True,
        'metadata': {}, 'answer': 'x', 'answer_aliases': ['x'], 'source_split': 'dev'})
    result = split_cohort(pd.DataFrame(rows), 11)
    nq = result[result.domain.eq('natural_questions')]
    assert nq.groupby('split_group').split.nunique().max() == 1
    assert result.set_index('example_id').loc['t', 'split'] == 'excluded_duplicate'


def test_verified_dataset_cache_loads_without_raw_corpora(tmp_path):
    cfg = config(tmp_path)
    cfg['paths']['nq'] = tmp_path / 'data/raw/natural_questions/missing.jsonl'
    cfg['paths']['techqa'] = tmp_path / 'data/raw/techqa'
    tables = {
        'qa': cohort_frame(),
        'docs': pd.DataFrame([{'document_id': 'd1', 'domain': 'natural_questions',
                               'title': 'Alpha', 'text': 'Paris'}]),
        'cohort': cohort_frame(),
    }
    checksums = {}
    for name, frame in tables.items():
        checksums[name] = file_hash(save_table(cfg, name, frame, processed=True))
    atomic_json(cfg['paths']['processed'] / 'dataset_manifest.json',
                {'table_sha256': checksums})

    qa, docs, cohort = prepare_data(cfg)

    assert len(qa) == 2
    assert len(docs) == 1
    assert len(cohort) == 2


def test_query_transformer_uses_source_training_only_and_excludes_self_match():
    training = pd.DataFrame({'example_id': ['a', 'b'], 'domain': ['natural_questions'] * 2,
        'split': ['train'] * 2, 'question': ['alpha source phrase', 'beta source phrase'],
        'source_split': ['train'] * 2})
    transformer = SourceQueryTransformer().fit(training)
    assert 'targetword' not in transformer.vectorizer.vocabulary_
    output = transformer.transform(pd.DataFrame({'example_id': ['a', 'z'],
        'domain': ['natural_questions', 'techqa'], 'split': ['train', 'techqa_test'],
        'source_split': ['train', 'dev'], 'question': ['alpha source phrase', 'targetword only']}))
    assert output.loc[0, 'nearest_train_similarity'] < 1
    assert output.loc[1, 'nearest_train_similarity'] == 0
    with pytest.raises(ValueError):
        SourceQueryTransformer().fit(training.assign(domain='techqa'))


def test_prepare_request_identity_changes_with_context_and_has_no_sampling_controls(tmp_path):
    cfg = config(tmp_path)
    cohort = cohort_frame().iloc[:1]
    docs = pd.DataFrame([{'document_id': 'd1', 'domain': 'natural_questions',
                          'title': 'Alpha', 'text': 'Paris is the reference.'}])
    plan1 = prepare_requests(cfg, cohort, docs, rankings_frame())
    requests = read_json(cfg['paths']['artifacts'] / 'generation/requests.json')
    assert len(plan1) == 3
    assert all('temperature' not in r['body'] and 'top_p' not in r['body'] for r in requests)
    assert all(r['body']['reasoning'] == {'effort': 'none'} for r in requests)
    assert all(r['body']['prompt_cache_options'] == {'mode': 'explicit'} for r in requests)
    original = {r['condition']: r['custom_id'] for r in requests}
    (cfg['paths']['artifacts'] / 'generation/manifest.json').unlink()
    docs.loc[0, 'text'] = 'Changed context.'
    prepare_requests(cfg, cohort, docs, rankings_frame())
    changed = {r['condition']: r['custom_id'] for r in read_json(
        cfg['paths']['artifacts'] / 'generation/requests.json')}
    assert changed['direct'] == original['direct']
    assert changed['hybrid_rrf'] != original['hybrid_rrf']


class Box:
    def __init__(self, **values):
        self.__dict__.update(values)

    def model_dump(self, mode='json'):
        return dict(self.__dict__)


class FakeFiles:
    def __init__(self):
        self.contents = {}

    def create(self, file, purpose):
        self.input = file.read().decode()
        return Box(id='file-in')

    def content(self, file_id):
        return Box(text=self.contents[file_id])


class FakeBatches:
    def __init__(self):
        self.batch = None
        self.created = 0

    def create(self, **kwargs):
        self.created += 1
        batch_id = f'batch-{self.created}'
        self.batch = {'id': batch_id, 'status': 'in_progress', 'input_file_id': kwargs['input_file_id'],
                      'metadata': kwargs['metadata']}
        return Box(id=batch_id, status='in_progress')

    def retrieve(self, batch_id):
        return Box(**self.batch)


class FakeClient:
    def __init__(self):
        self.files, self.batches = FakeFiles(), FakeBatches()


def response_record(request, answer='Paris'):
    return {'custom_id': request['custom_id'], 'response': {'status_code': 200, 'request_id': 'req-1',
        'body': {'id': 'resp-1', 'status': 'completed', 'model': 'gpt-5.6-luna',
            'service_tier': 'default', 'output': [{'type': 'message', 'content': [
                {'type': 'output_text', 'text': answer}]}],
            'usage': {'input_tokens': 20, 'output_tokens': 4,
                      'input_tokens_details': {'cached_tokens': 0}}}}}


def test_batch_resume_matches_unordered_results_and_preserves_failures(tmp_path):
    cfg = config(tmp_path)
    cohort = cohort_frame().iloc[:1]
    docs = pd.DataFrame([{'document_id': 'd1', 'domain': 'natural_questions',
                          'title': 'Alpha', 'text': 'Paris is the reference.'}])
    prepare_requests(cfg, cohort, docs, rankings_frame())
    client = FakeClient()
    submitted = batch_step(cfg, 'submit', client=client, request_limit=3)
    assert len(submitted['active_batches']) == 1
    requests = read_json(cfg['paths']['artifacts'] / 'generation/requests.json')
    records = [response_record(requests[2]),
               {'custom_id': requests[1]['custom_id'], 'error': {'code': 'rate_limit_exceeded', 'message': 'later'}},
               response_record(requests[0])]
    client.files.contents['out'] = '\n'.join(json.dumps(r) for r in records)
    client.batches.batch.update(status='completed', output_file_id='out', error_file_id=None)
    status = batch_step(cfg, 'collect', client=client)
    table = result_table(cfg).set_index('custom_id')
    assert status['complete'] is False
    assert table.loc[requests[0]['custom_id'], 'status'] == 'completed'
    assert table.loc[requests[1]['custom_id'], 'error_code'] == 'rate_limit_exceeded'
    # Default submission never silently retries the failed request.
    assert generation_status(cfg)['reserved_usd_all_attempts'] > 0
    assert not (cfg['paths']['artifacts'] / 'generation/submission.lock').exists()


def test_batch_budget_guard_persists_before_network_submission(tmp_path):
    cfg = config(tmp_path)
    cohort = cohort_frame().iloc[:1]
    docs = pd.DataFrame([{'document_id': 'd1', 'domain': 'natural_questions',
                          'title': 'Alpha', 'text': 'Paris'}])
    prepare_requests(cfg, cohort, docs, rankings_frame())
    cfg['generation']['budget_usd'] = 0
    with pytest.raises(RuntimeError, match='spending guard'):
        batch_step(cfg, 'submit', client=FakeClient())


def test_finished_batch_run_with_saved_failure_does_not_require_api_client(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cohort = cohort_frame().iloc[:1]
    docs = pd.DataFrame([{'document_id': 'd1', 'domain': 'natural_questions',
                          'title': 'Alpha', 'text': 'Paris'}])
    prepare_requests(cfg, cohort, docs, rankings_frame())
    requests = read_json(cfg['paths']['artifacts'] / 'generation/requests.json')
    for index, request in enumerate(requests):
        atomic_json(cfg['paths']['artifacts'] / 'generation/results' /
                    f"{request['custom_id']}.json",
                    {'custom_id': request['custom_id'],
                     'status': 'failed' if index == 0 else 'completed'})
    monkeypatch.setattr('adaptive_rag.luna_v2.generation._client',
                        lambda cfg: (_ for _ in ()).throw(AssertionError('client requested')))

    status = run_batches(cfg)

    assert status['finished'] is True
    assert status['complete'] is False


def test_bounded_parallel_submission_uses_distinct_pending_requests(tmp_path):
    cfg = config(tmp_path)
    cohort = cohort_frame().iloc[:1]
    docs = pd.DataFrame([{'document_id': 'd1', 'domain': 'natural_questions',
                          'title': 'Alpha', 'text': 'Paris'}])
    prepare_requests(cfg, cohort, docs, rankings_frame())
    client = FakeClient()
    first = batch_step(cfg, 'submit', client=client, request_limit=1)
    second = batch_step(cfg, 'submit', client=client, request_limit=1,
                        allow_additional_active=True)
    assert len(first['active_batches']) == 1
    assert len(second['active_batches']) == 2
    ledger = read_json(cfg['paths']['artifacts'] / 'generation/ledger.json')
    assert ledger['attempts'][0]['request_ids'] != ledger['attempts'][1]['request_ids']
    assert {attempt['batch_id'] for attempt in ledger['attempts']} == {'batch-1', 'batch-2'}


def test_explicit_retry_never_duplicates_request_already_in_active_batch(tmp_path):
    cfg = config(tmp_path)
    cohort = cohort_frame().iloc[:1]
    docs = pd.DataFrame([{'document_id': 'd1', 'domain': 'natural_questions',
                          'title': 'Alpha', 'text': 'Paris'}])
    prepare_requests(cfg, cohort, docs, rankings_frame())
    requests = read_json(cfg['paths']['artifacts'] / 'generation/requests.json')
    atomic_json(cfg['paths']['artifacts'] / 'generation/results' /
                f"{requests[0]['custom_id']}.json",
                {'custom_id': requests[0]['custom_id'], 'status': 'failed',
                 'error_code': 'unusable_response'})
    client = FakeClient()
    batch_step(cfg, 'submit', client=client, retry_failed=True, request_limit=1)
    batch_step(cfg, 'submit', client=client, retry_failed=True, request_limit=1,
               allow_additional_active=True)
    ledger = read_json(cfg['paths']['artifacts'] / 'generation/ledger.json')
    assert ledger['attempts'][0]['request_ids'] == [requests[0]['custom_id']]
    assert ledger['attempts'][1]['request_ids'] == [requests[1]['custom_id']]


def test_parse_response_requires_exact_model_completion_and_usage(tmp_path):
    cfg = config(tmp_path)
    request = {'custom_id': 'abc', 'body': {'model': 'gpt-5.6-luna'}}
    good = parse_batch_result(cfg, request, response_record(request), 'batch')
    assert good['status'] == 'completed' and good['generation_latency_ms'] is None
    bad = response_record(request)
    bad['response']['body']['model'] = 'another-model'
    assert parse_batch_result(cfg, request, bad, 'batch')['status'] == 'failed'


def test_evaluation_separates_accuracy_task_success_and_route_cost(tmp_path):
    cfg = config(tmp_path)
    cohort = cohort_frame()
    rows = []
    for item in cohort.itertuples():
        for condition in ('direct', 'hybrid_rrf', 'hybrid_rrf_reranked'):
            answer = 'Paris' if item.answerable else ABSTAIN
            rows.append({'custom_id': item.example_id + condition, 'example_id': item.example_id,
                'domain': item.domain, 'condition': condition, 'status': 'completed',
                'generated_answer': answer, 'estimated_cost_usd': .001,
                'generation_latency_ms': None})
    evaluated = evaluate_results(cfg, cohort, pd.DataFrame(rows))
    require_complete(evaluated)
    nq = evaluated[evaluated.example_id.eq('nq1')]
    tq = evaluated[evaluated.example_id.eq('tq1')]
    assert nq.answer_correct.all() and nq.task_success.all()
    assert not tq.answer_correct.any() and tq.task_success.all() and not tq.answered.any()
    actions = {'nq1': 'ABSTAIN', 'tq1': 'ABSTAIN'}
    replay = replay_actions(cfg, evaluated, rankings_frame(), actions, system='adaptive').set_index('example_id')
    assert replay.loc['nq1', 'answer_correct'] == False
    assert replay.loc['nq1', 'task_success'] == False
    assert replay.loc['tq1', 'task_success'] == True
    assert (replay.retrieval_latency_ms == 4).all()
    assert replay.generation_latency_ms.isna().all()


def test_failed_condition_excludes_whole_example_from_paired_analysis(tmp_path):
    cfg = config(tmp_path)
    cohort = cohort_frame()
    rows = []
    for item in cohort.itertuples():
        for condition in ('direct', 'hybrid_rrf', 'hybrid_rrf_reranked'):
            status = 'failed' if item.example_id == 'tq1' and condition == 'direct' else 'completed'
            rows.append({'custom_id': item.example_id + condition, 'example_id': item.example_id,
                'domain': item.domain, 'condition': condition, 'status': status,
                'generated_answer': 'Paris' if status == 'completed' else None,
                'estimated_cost_usd': .001, 'generation_latency_ms': None})
    evaluated = evaluate_results(cfg, cohort, pd.DataFrame(rows))
    paired, exclusions = paired_complete_cases(evaluated)
    assert set(paired.example_id) == {'nq1'}
    assert set(paired.condition) == {'direct', 'hybrid_rrf', 'hybrid_rrf_reranked'}
    assert exclusions[['example_id', 'unavailable_conditions']].to_dict('records') == [
        {'example_id': 'tq1', 'unavailable_conditions': 'direct'}]


def test_noncanonical_abstention_phrase_is_an_answer(tmp_path):
    cfg = config(tmp_path)
    cohort = cohort_frame().iloc[1:]
    rows = [{'custom_id': str(i), 'example_id': 'tq1', 'domain': 'techqa', 'condition': condition,
             'status': 'completed', 'generated_answer': 'unknown setting',
             'estimated_cost_usd': .001, 'generation_latency_ms': None}
            for i, condition in enumerate(('direct', 'hybrid_rrf', 'hybrid_rrf_reranked'))]
    evaluated = evaluate_results(cfg, cohort, pd.DataFrame(rows))
    assert evaluated.answered.all()
    assert not evaluated.task_success.any()


def test_policy_and_final_report_execute_on_frozen_synthetic_cohort(tmp_path):
    cfg = config(tmp_path)
    cfg['policy']['minimum_class_examples'] = 2
    cohort, rankings, evaluated = [], [], []
    conditions = ('direct', 'hybrid_rrf', 'hybrid_rrf_reranked')
    for i in range(36):
        split = 'train' if i < 22 else 'validation' if i < 29 else 'nq_test'
        example_id = f'n{i}'
        action_type = i % 4
        cohort.append({'example_id': example_id, 'domain': 'natural_questions',
            'question': f'Where is source item {i}?', 'answer': f'answer{i}',
            'answer_aliases': [f'answer{i}'], 'answerable': True,
            'gold_document_ids': [f'd{i}'], 'source_split': 'train', 'metadata': '{}',
            'annotation_category': 'short_or_yes_no', 'generation_eligible': True,
            'split': split, 'split_group': f'g{i}', 'cross_domain_duplicate': False})
        for method, latency in [('hybrid_rrf', 2.), ('hybrid_rrf_reranked', 5.)]:
            rankings.append({'example_id': example_id, 'domain': 'natural_questions',
                'retrieval_method': method, 'document_ids': [f'd{i}'], 'scores': [1., .2],
                'latency_ms': latency, 'metadata': '{}'})
        for j, condition in enumerate(conditions):
            correct = action_type < 3 and j >= action_type
            evaluated.append({'custom_id': example_id + condition, 'example_id': example_id,
                'domain': 'natural_questions', 'condition': condition, 'status': 'completed',
                'generated_answer': f'answer{i}' if correct else 'wrong', 'estimated_cost_usd': .0001 * (j + 1),
                'generation_latency_ms': np.nan, 'answerable': True, 'split': split,
                'exact_match': correct, 'token_f1': float(correct), 'answer_correct': correct,
                'task_success': correct, 'abstained': False, 'answered': True})
    for i in range(8):
        example_id = f't{i}'
        cohort.append({'example_id': example_id, 'domain': 'techqa', 'question': f'Target issue {i}?',
            'answer': '' if i % 2 else f'target{i}', 'answer_aliases': [] if i % 2 else [f'target{i}'],
            'answerable': not i % 2, 'gold_document_ids': [], 'source_split': 'dev', 'metadata': '{}',
            'annotation_category': 'answerable' if not i % 2 else 'benchmark_unanswerable',
            'generation_eligible': True, 'split': 'techqa_test', 'split_group': example_id,
            'cross_domain_duplicate': False})
        for method, latency in [('hybrid_rrf', 3.), ('hybrid_rrf_reranked', 7.)]:
            rankings.append({'example_id': example_id, 'domain': 'techqa', 'retrieval_method': method,
                'document_ids': [f'td{i}'], 'scores': [1., .3], 'latency_ms': latency, 'metadata': '{}'})
        for j, condition in enumerate(conditions):
            answerable = not i % 2
            evaluated.append({'custom_id': example_id + condition, 'example_id': example_id,
                'domain': 'techqa', 'condition': condition, 'status': 'completed',
                'generated_answer': f'target{i}' if answerable else ABSTAIN,
                'estimated_cost_usd': .0001 * (j + 1), 'generation_latency_ms': np.nan,
                'answerable': answerable, 'split': 'techqa_test', 'exact_match': answerable,
                'token_f1': 1. if answerable else np.nan, 'answer_correct': answerable,
                'task_success': True, 'abstained': not answerable, 'answered': answerable})
    cohort, rankings, evaluated = map(pd.DataFrame, (cohort, rankings, evaluated))
    save_table(cfg, 'cohort', cohort, processed=True)
    save_table(cfg, 'rankings', rankings)
    save_table(cfg, 'evaluated_generations', evaluated)
    atomic_json(cfg['paths']['artifacts'] / 'generation/manifest.json', {'signature': 'synthetic'})
    bundle, diagnostics, trials = train_policy(cfg, cohort, rankings, evaluated)
    replay, summary, intervals, drift = final_evaluation(cfg, bundle, cohort, rankings, evaluated)
    assert set(diagnostics.stage) == {'continue_after_query', 'continue_after_hybrid', 'escalate_or_abstain'}
    assert len(trials) == 125
    assert set(replay.system) == {'adaptive', 'DIRECT', 'RETRIEVE', 'ESCALATE'}
    assert set(summary.domain) == {'natural_questions', 'techqa'}
    assert not intervals.empty and not drift.empty
    assert (cfg['paths']['artifacts'] / 'blind_answer_audit.csv').exists()
