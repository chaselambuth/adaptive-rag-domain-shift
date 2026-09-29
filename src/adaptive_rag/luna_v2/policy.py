# Source-only feature transforms and validation-selected Luna routing policy.

from __future__ import annotations

from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.tree import DecisionTreeClassifier

from adaptive_rag.query_features import extract_query_features
from .common import atomic_json, digest, file_hash, read_json, save_table, versions
from .evaluation import oracle_actions, replay_actions, require_complete
from .retrieval import hybrid_features

QUERY_COLUMNS = ['question_char_length', 'question_token_length', 'query_lexical_diversity',
    'punctuation_density', 'code_symbol_density', 'numeric_token_density', 'technical_term_density',
    'nq_centroid_similarity', 'nearest_train_similarity']
HYBRID_COLUMNS = ['hybrid_top1', 'hybrid_margin', 'hybrid_mean', 'hybrid_std', 'hybrid_entropy']


# Fit vocabulary and source-reference similarities on NQ training text only.
class SourceQueryTransformer:
    # Learn the source vocabulary, centroid, and neighbours from training text.
    def fit(self, training):
        if set(training.domain) != {'natural_questions'} or set(training.split) != {'train'}:
            raise ValueError('Query transformer must fit only NQ training examples.')
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(training.question)
        self.centroid = normalize(np.asarray(self.matrix.mean(axis=0)))
        self.texts = training.question.str.lower().str.split().str.join(' ').to_numpy()
        self.train_ids = training.example_id.tolist()
        return self

    # Project questions into the frozen source feature space.
    def transform(self, qa):
        result = extract_query_features(qa)
        matrix = self.vectorizer.transform(qa.question)
        result['nq_centroid_similarity'] = np.asarray(matrix @ self.centroid.T).ravel()
        similarities = (matrix @ self.matrix.T).toarray()
        for i, text in enumerate(qa.question.str.lower().str.split().str.join(' ')):
            similarities[i, self.texts == text] = 0.  # Same rule offline and at inference.
        result['nearest_train_similarity'] = similarities.max(axis=1)
        return result


# Provide a fixed-probability fallback for unsupported routing stages.
class ConstantProbability:
    def __init__(self, probability):
        self.probability = float(probability)

    # Return the same two-class probability for every input row.
    def predict_proba(self, frame):
        p = np.full(len(frame), self.probability)
        return np.column_stack([1 - p, p])


# Combine source-only query features with pre-escalation retrieval signals.
def feature_frame(transformer, cohort, rankings):
    query = transformer.transform(cohort)
    return query.merge(hybrid_features(rankings), on='example_id', how='left', validate='one_to_one')


# Evaluate the three fitted stage models on one feature table.
def stage_probabilities(bundle, features):
    return np.column_stack([asset['model'].predict_proba(features[asset['columns']])[:, 1]
                            for asset in bundle['stages']])


# Convert stage probabilities and thresholds into final routing actions.
def choose_actions(probabilities, thresholds):
    gate1, gate2, gate3 = [probabilities[:, i] >= thresholds[i] for i in range(3)]
    return np.where(~gate1, 'DIRECT', np.where(~gate2, 'RETRIEVE', np.where(gate3, 'ESCALATE', 'ABSTAIN')))


# Hash all inputs that define the frozen Luna policy.
def _signature(cfg):
    return digest({'generation': read_json(cfg['paths']['artifacts'] / 'generation/manifest.json')['signature'],
                   'evaluated': file_hash(cfg['paths']['artifacts'] / 'evaluated_generations.parquet'),
                   'rankings': file_hash(cfg['paths']['artifacts'] / 'rankings.parquet'),
                   'cohort': file_hash(cfg['paths']['processed'] / 'cohort.parquet'),
                   'policy': cfg['policy'], 'seed': cfg['seed'], 'version': cfg['version']})


# Fit source-only stages and select thresholds on NQ validation data.
def train_policy(cfg, cohort, rankings, evaluated):
    require_complete(evaluated)
    cohort = cohort[~cohort.split.eq('excluded_duplicate')].reset_index(drop=True)
    oracle = oracle_actions(evaluated)
    source_train = cohort[cohort.split.eq('train')]
    transformer = SourceQueryTransformer().fit(source_train)
    features = feature_frame(transformer, cohort, rankings).merge(
        oracle[['example_id', 'oracle_action', 'split']], on='example_id', validate='one_to_one')
    train = features[features.split.eq('train')]
    validation = features[features.split.eq('validation')]
    if validation.empty or train.empty:
        raise ValueError('Training and validation splits must both be nonempty.')
    specifications = [
        ('continue_after_query', QUERY_COLUMNS, lambda f: np.ones(len(f), dtype=bool), lambda f: f.oracle_action.ne('DIRECT')),
        ('continue_after_hybrid', QUERY_COLUMNS + HYBRID_COLUMNS, lambda f: f.oracle_action.ne('DIRECT'),
         lambda f: f.oracle_action.isin(['ESCALATE', 'ABSTAIN'])),
        ('escalate_or_abstain', QUERY_COLUMNS + HYBRID_COLUMNS, lambda f: f.oracle_action.isin(['ESCALATE', 'ABSTAIN']),
         lambda f: f.oracle_action.eq('ESCALATE'))]
    stages, diagnostics, probability_bins = [], [], []
    for name, columns, eligible, target in specifications:
        t, v = train[eligible(train)], validation[eligible(validation)]
        y, vy = target(t).astype(int), target(v).astype(int)
        positives, negatives = int(y.sum()), int(len(y) - y.sum())
        candidates = {'constant': ConstantProbability(float(y.mean()) if len(y) else 0.)}
        if min(positives, negatives) >= cfg['policy']['minimum_class_examples']:
            candidates['logistic'] = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                LogisticRegression(C=1., max_iter=2000, random_state=cfg['seed']))
            candidates['tree'] = make_pipeline(SimpleImputer(strategy='median'),
                DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=cfg['seed']))
            for key in ('logistic', 'tree'):
                candidates[key].fit(t[columns], y)
        scores = {}
        for key, model in candidates.items():
            scores[key] = float(np.mean((model.predict_proba(v[columns])[:, 1] - vy.to_numpy()) ** 2)) if len(v) else None
        winner = min(scores, key=lambda key: (scores[key] if scores[key] is not None else 0., key != 'constant'))
        if len(v):
            selected_probability = candidates[winner].predict_proba(v[columns])[:, 1]
            edges = np.linspace(0, 1, 6)
            for lower, upper in zip(edges[:-1], edges[1:]):
                mask = ((selected_probability >= lower) &
                        (selected_probability <= upper if upper == 1 else selected_probability < upper))
                if mask.any():
                    probability_bins.append({'stage': name, 'bin_lower': lower, 'bin_upper': upper,
                        'examples': int(mask.sum()),
                        'mean_predicted_probability': float(selected_probability[mask].mean()),
                        'observed_positive_rate': float(vy.to_numpy()[mask].mean())})
        stages.append({'name': name, 'columns': columns, 'model': candidates[winner], 'model_name': winner})
        diagnostics.append({'stage': name, 'training_examples': len(t), 'positive_examples': positives,
            'negative_examples': negatives, 'validation_examples': len(v), 'selected_model': winner,
            'validation_brier': scores[winner], 'candidate_brier': str(scores),
            'rare_class_fallback': len(candidates) == 1})
    bundle = {'version': cfg['version'], 'signature': _signature(cfg), 'transformer': transformer,
        'stages': stages, 'train_ids': source_train.example_id.tolist(),
        'validation_ids': validation.example_id.tolist(), 'thresholds': None}
    probabilities = stage_probabilities(bundle, validation)
    ids = validation.example_id.tolist()
    # Precompute the four possible outcomes. Search thresholds only on NQ validation.
    outcomes = {action: replay_actions(cfg, evaluated, rankings, dict.fromkeys(ids, action), system=action)
                .set_index('example_id').loc[ids] for action in ('DIRECT', 'RETRIEVE', 'ESCALATE', 'ABSTAIN')}
    trials = []
    for thresholds in product((0., .25, .5, .75, 1.01), repeat=3):
        actions = choose_actions(probabilities, thresholds)
        objective = np.zeros(len(ids))
        for action, table in outcomes.items():
            mask = actions == action
            value = table.task_utility - cfg['policy']['cost_weight'] * table.estimated_cost_usd
            objective[mask] = value.to_numpy()[mask]
        trials.append({'threshold_1': thresholds[0], 'threshold_2': thresholds[1], 'threshold_3': thresholds[2],
                       'validation_objective': float(objective.mean())})
    trials = pd.DataFrame(trials).sort_values(['validation_objective', 'threshold_1', 'threshold_2', 'threshold_3'],
                                             ascending=[False, True, True, True], kind='stable')
    best = trials.iloc[0]
    bundle['thresholds'] = [float(best[f'threshold_{i}']) for i in (1, 2, 3)]
    bundle['versions'] = versions()
    # Freeze before any NQ-test or TechQA outcome summaries are inspected.
    joblib.dump(bundle, cfg['paths']['artifacts'] / 'frozen_policy.joblib')
    atomic_json(cfg['paths']['artifacts'] / 'policy_manifest.json', {
        'signature': bundle['signature'], 'thresholds': bundle['thresholds'], 'versions': bundle['versions'],
        'training_examples': len(train), 'validation_examples': len(validation), 'diagnostics': diagnostics,
        'objective': 'mean task utility - cost_weight * estimated Batch USD; validation only',
        'label_note': 'ABSTAIN oracle on answerable all-fail cases is a best-observed-action proxy, not unanswerability.',
        'features': {a['name']: a['columns'] for a in stages}})
    save_table(cfg, 'policy_training_diagnostics', pd.DataFrame(diagnostics))
    save_table(cfg, 'validation_probability_bins', pd.DataFrame(probability_bins))
    save_table(cfg, 'validation_threshold_search', trials)
    save_table(cfg, 'policy_features', features)
    save_table(cfg, 'oracle_actions', oracle)
    return bundle, pd.DataFrame(diagnostics), trials


# Load the frozen policy and reject mismatched experiment artifacts.
def load_policy(cfg):
    bundle = joblib.load(cfg['paths']['artifacts'] / 'frozen_policy.joblib')
    if bundle['signature'] != _signature(cfg):
        raise ValueError('Frozen policy does not match this run. Rebuild notebook 04.')
    return bundle


# Generate actions and stage probabilities for a cohort.
def predict_actions(bundle, cohort, rankings):
    features = feature_frame(bundle['transformer'], cohort, rankings)
    probabilities = stage_probabilities(bundle, features)
    return dict(zip(features.example_id, choose_actions(probabilities, bundle['thresholds'])))
