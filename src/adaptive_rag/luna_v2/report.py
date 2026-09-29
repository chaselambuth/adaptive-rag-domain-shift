# Held-out evaluation exports, drift summaries, and blind-audit materials.

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import atomic_json, save_table
from .evaluation import metric_summary, paired_bootstrap, replay_actions, require_complete
from .policy import feature_frame, predict_actions


# Evaluate the frozen policy and export metrics, drift, and blind-audit files.
def final_evaluation(cfg, bundle, cohort, rankings, evaluated):
    require_complete(evaluated)
    eligible_ids = set(evaluated.example_id)
    heldout = cohort[cohort.split.isin(['nq_test', 'techqa_test']) &
                     cohort.example_id.isin(eligible_ids)].copy()
    ids = heldout.example_id.tolist()
    if set(ids) & (set(bundle['train_ids']) | set(bundle['validation_ids'])):
        raise ValueError('Evaluation overlaps policy development data.')
    pieces = [replay_actions(cfg, evaluated, rankings, predict_actions(bundle, heldout, rankings), system='adaptive')]
    for action in ('DIRECT', 'RETRIEVE', 'ESCALATE'):
        pieces.append(replay_actions(cfg, evaluated, rankings, dict.fromkeys(ids, action), system=action))
    rows = pd.concat(pieces, ignore_index=True)
    summary = metric_summary(rows)
    intervals = paired_bootstrap(rows, heldout.set_index('example_id').split_group.to_dict(), seed=cfg['seed'])
    features = feature_frame(bundle['transformer'], cohort[~cohort.split.eq('excluded_duplicate')], rankings)
    feature_columns = bundle['stages'][-1]['columns']
    train_features = features[features.example_id.isin(bundle['train_ids'])]
    drift_rows = []
    for domain, group in heldout.groupby('domain'):
        test_features = features[features.example_id.isin(group.example_id)]
        for column in feature_columns:
            scale = float(train_features[column].std(ddof=0))
            shift = float(test_features[column].mean() - train_features[column].mean())
            drift_rows.append({'domain': domain, 'feature': column, 'train_mean': train_features[column].mean(),
                'test_mean': test_features[column].mean(), 'standardized_mean_shift': shift / scale if scale > 0 else np.nan})
    drift = pd.DataFrame(drift_rows)
    for name, table in (('final_replay', rows), ('final_metrics', summary), ('paired_intervals', intervals), ('feature_drift', drift)):
        save_table(cfg, name, table)
        table.to_csv(cfg['paths']['artifacts'] / (name + '.csv'), index=False)
    # Blind annotation sheet: route, model, and automatic scores are kept in a separate key.
    sample_parts = [g.sample(min(20, len(g)), random_state=cfg['seed'])
                    for _, g in heldout.groupby(['domain', 'answerable'])]
    sampled_ids = pd.concat(sample_parts).example_id
    answers = evaluated[evaluated.example_id.isin(sampled_ids)].sample(frac=1., random_state=cfg['seed']).copy()
    answers['audit_id'] = [f'luna-audit-{i:04d}' for i in range(len(answers))]
    blind = answers[['audit_id', 'example_id', 'generated_answer']].merge(
        heldout[['example_id', 'question', 'answer', 'answer_aliases']], on='example_id')
    for column in ('human_correct', 'human_abstention_appropriate', 'notes'):
        blind[column] = ''
    blind.drop(columns='example_id').to_csv(cfg['paths']['artifacts'] / 'blind_answer_audit.csv', index=False)
    answers[['audit_id', 'example_id', 'condition', 'answer_correct', 'task_success']].to_csv(
        cfg['paths']['artifacts'] / 'blind_answer_audit_key.csv', index=False)
    atomic_json(cfg['paths']['artifacts'] / 'final_report_manifest.json', {
        'policy_signature': bundle['signature'], 'heldout_examples': len(heldout),
        'domain_counts': heldout.domain.value_counts().to_dict(),
        'deployment_status': 'research_only_pending_human_quality_review_and_online_latency_validation',
        'limitations': ['TechQA lexical answer accuracy is a proxy; complete the blind review.',
            'Batch generation latency is unavailable; retrieval time is not end-to-end latency.',
            'Thresholds and model selection used only NQ validation, with small source samples.',
            'Budget/cost figures are estimates, not an invoice or provider-enforced hard cap.',
            'No calibrated-confidence claim; rare classes may require constant stage rules.',
            'NQ negatives and long-only records are excluded from primary generation.',
            'The frozen policy remains comparison-only; static Hybrid RRF is the strongest baseline.']})
    return rows, summary, intervals, drift
