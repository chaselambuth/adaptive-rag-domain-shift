# Paired availability, strategy replay, metrics, and uncertainty estimates.

from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_rag.answer_evaluation import collect_references, exact_match, normalize_answer, token_f1
from .common import ABSTAIN, CONDITIONS, save_table

ACTION_CONDITION = {'DIRECT': 'direct', 'RETRIEVE': 'hybrid_rrf', 'ESCALATE': 'hybrid_rrf_reranked'}


# Score completed generations and mark unavailable rows explicitly.
def evaluate_results(cfg, cohort, results):
    expected = cohort[~cohort.split.eq('excluded_duplicate')].set_index('example_id')
    if results.duplicated(['example_id', 'condition']).any():
        raise ValueError('Duplicate generation results.')
    if set(zip(results.example_id, results.condition)) != {(i, c) for i in expected.index for c in CONDITIONS}:
        raise ValueError('Generation results do not match the frozen cohort and all three conditions.')
    rows = []
    for result in results.to_dict('records'):
        source = expected.loc[result['example_id']]
        result.update(answerable=bool(source.answerable), split=source.split)
        if result['status'] != 'completed':
            result.update(exact_match=np.nan, token_f1=np.nan, answer_correct=np.nan,
                          task_success=np.nan, abstained=np.nan, answered=np.nan)
        else:
            answer = str(result['generated_answer'])
            abstained = normalize_answer(answer) == normalize_answer(ABSTAIN)
            references = collect_references(source)
            em = exact_match(answer, references) if source.answerable else False
            f1 = token_f1(answer, references) if source.answerable else np.nan
            correct = bool(source.answerable and not abstained and (em or f1 >= cfg['generation']['min_f1_for_correct']))
            result.update(exact_match=em, token_f1=f1, answer_correct=correct,
                task_success=correct if source.answerable else abstained,
                abstained=abstained, answered=not abstained)
        rows.append(result)
    evaluated = pd.DataFrame(rows)
    eligibility = evaluated.groupby('example_id').agg(
        all_conditions=('condition', lambda values: set(values) == set(CONDITIONS)),
        all_completed=('status', lambda values: values.eq('completed').all()))
    eligible = eligibility.all(axis=1)
    evaluated['analysis_eligible'] = evaluated.example_id.map(eligible).astype(bool)
    save_table(cfg, 'evaluated_generations', evaluated)
    return evaluated


# Keep examples with all conditions available and report each exclusion.
def paired_complete_cases(evaluated):
    if evaluated.duplicated(['example_id', 'condition']).any():
        raise ValueError('Duplicate generation results.')
    condition_sets = evaluated.groupby('example_id').condition.agg(set)
    if not condition_sets.map(lambda values: values == set(CONDITIONS)).all():
        raise ValueError('Every example must retain a recorded row for all three conditions.')
    eligible = evaluated.groupby('example_id').status.agg(lambda values: values.eq('completed').all())
    eligible_ids = set(eligible[eligible].index)
    paired = evaluated[evaluated.example_id.isin(eligible_ids)].copy().reset_index(drop=True)
    if paired.empty:
        raise ValueError('No examples have complete paired generation results.')
    unavailable = evaluated[~evaluated.status.eq('completed')].copy()
    exclusions = (unavailable.groupby('example_id', as_index=False)
        .agg(domain=('domain', 'first'), split=('split', 'first'),
             unavailable_conditions=('condition', lambda values: ','.join(sorted(values))),
             unavailable_statuses=('status', lambda values: ','.join(sorted(set(values))))))
    require_complete(paired)
    return paired, exclusions


# Reject downstream analysis when a paired example is incomplete.
def require_complete(evaluated):
    missing = evaluated[~evaluated.status.eq('completed')]
    if len(missing):
        raise ValueError(f'{len(missing)} generation requests are unavailable. Use paired_complete_cases before '
                         'analysis; unavailable outputs are never scored as wrong or silently dropped.')
    counts = evaluated.groupby('example_id').condition.agg(lambda s: set(s))
    if not counts.map(lambda s: s == set(CONDITIONS)).all():
        raise ValueError('All three conditions must cover exactly the same examples.')


# Assign the cheapest successful observed action for policy training.
def oracle_actions(evaluated):
    require_complete(evaluated)
    rows = []
    for example_id, group in evaluated.groupby('example_id'):
        first = group.iloc[0]
        action = 'ABSTAIN'
        if first.answerable:
            success = group.set_index('condition').answer_correct.to_dict()
            for candidate, condition in ACTION_CONDITION.items():
                if success[condition]:
                    action = candidate
                    break
        rows.append({'example_id': example_id, 'domain': first.domain, 'split': first.split,
                     'oracle_action': action, 'answerable': bool(first.answerable)})
    return pd.DataFrame(rows)


# Replay selected actions into comparable answer, cost, and latency rows.
def replay_actions(cfg, evaluated, rankings, actions, *, system):
    require_complete(evaluated)
    by_result = evaluated.set_index(['example_id', 'condition'])
    by_rank = rankings.set_index(['example_id', 'retrieval_method'])
    rows = []
    for example_id, action in actions.items():
        if action not in (*ACTION_CONDITION, 'ABSTAIN'):
            raise ValueError('Unknown policy action.')
        first = by_result.loc[(example_id, 'direct')]
        if action == 'ABSTAIN':
            answered, correct, success, cost = False, False, not bool(first.answerable), 0.
            answer = ABSTAIN
            retrieval_ms = by_rank.loc[(example_id, 'hybrid_rrf'), 'latency_ms']
        else:
            condition = ACTION_CONDITION[action]
            result = by_result.loc[(example_id, condition)]
            answered, correct, success = bool(result.answered), bool(result.answer_correct), bool(result.task_success)
            cost = float(result.estimated_cost_usd)
            answer = result.generated_answer
            retrieval_ms = 0. if action == 'DIRECT' else by_rank.loc[(example_id, condition), 'latency_ms']
        utility = 1. if success else -cfg['policy']['wrong_answer_penalty'] if answered else 0.
        rows.append({'example_id': example_id, 'domain': first.domain, 'split': first.split,
            'answerable': bool(first.answerable), 'system': system, 'action': action,
            'generated_answer': answer, 'answered': answered, 'answer_correct': correct,
            'task_success': success, 'task_utility': utility, 'estimated_cost_usd': cost,
            'retrieval_latency_ms': float(retrieval_ms), 'generation_latency_ms': np.nan})
    return pd.DataFrame(rows)


# Aggregate quality, abstention, utility, cost, and latency metrics.
def metric_summary(rows):
    output = []
    for (domain, system), group in rows.groupby(['domain', 'system'], sort=True):
        for stratum, subset in (('all', group), ('answerable', group[group.answerable]),
                                 ('unanswerable', group[~group.answerable])):
            if subset.empty:
                continue
            answerable = subset[subset.answerable]
            answered = subset[subset.answered]
            output.append({'domain': domain, 'system': system, 'stratum': stratum,
                'examples': len(subset), 'answerable_examples': len(answerable),
                'answer_accuracy': answerable.answer_correct.mean() if len(answerable) else np.nan,
                'task_success': subset.task_success.mean(), 'coverage': subset.answered.mean(),
                'selective_answer_accuracy': answered.answer_correct.mean() if len(answered) else np.nan,
                'unnecessary_abstention_rate': (~answerable.answered).mean() if len(answerable) else np.nan,
                'mean_utility': subset.task_utility.mean(),
                'mean_estimated_batch_usd': subset.estimated_cost_usd.mean(),
                'mean_retrieval_ms': subset.retrieval_latency_ms.mean()})
    return pd.DataFrame(output)


# Estimate paired uncertainty with frozen query/page-group resampling.
def paired_bootstrap(rows, groups, *, seed, repeats=2000):
    output = []
    rng = np.random.default_rng(seed)
    for domain, data in rows.groupby('domain'):
        for baseline in ('DIRECT', 'RETRIEVE', 'ESCALATE'):
            for metric in ('task_success', 'task_utility', 'estimated_cost_usd'):
                pivot = data[data.system.isin(['adaptive', baseline])].pivot(index='example_id', columns='system', values=metric)
                if pivot.isna().any().any():
                    raise ValueError('Unpaired bootstrap input.')
                differences = (pivot['adaptive'].astype(float) - pivot[baseline].astype(float)).to_frame('delta')
                differences['group'] = differences.index.map(groups)
                if differences.group.isna().any():
                    raise ValueError('Missing bootstrap group.')
                clusters = differences.groupby('group').delta.agg(['sum', 'count'])
                sums, counts = clusters['sum'].to_numpy(), clusters['count'].to_numpy()
                sample = rng.integers(len(clusters), size=(repeats, len(clusters)))
                values = sums[sample].sum(axis=1) / counts[sample].sum(axis=1)
                output.append({'domain': domain, 'baseline': baseline, 'metric': metric,
                    'examples': len(pivot), 'clusters': len(clusters), 'adaptive_minus_baseline': differences.delta.mean(),
                    'ci_low': float(np.quantile(values, .025)), 'ci_high': float(np.quantile(values, .975)),
                    'note': 'Fixed-model uncertainty over sampled query/page groups; no retraining uncertainty.'})
    return pd.DataFrame(output)
