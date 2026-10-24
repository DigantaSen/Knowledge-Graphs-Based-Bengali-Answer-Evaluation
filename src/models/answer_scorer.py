"""
answer_scorer.py
================
XGBoost + MLP ensemble answer scorer with feature engineering.
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.config import CV_N_SPLITS, RANDOM_STATE


class AnswerScorer:
    """
    Ensemble scorer combining XGBoost and an MLP Regressor.

    Input features (10):
      base_sim   — LaBSE sentence cosine similarity
      graph_sim  — Siamese GNN cosine similarity
      penalty    — Karak validation penalty (0..1)
      entity_mm  — Entity mismatch ratio (0..1)
      coverage   — Sentence coverage score (0..1)
      neg_mm     — Negation mismatch flag (0 or 1)
      wrong_num  — Wrong-number flag (0 or 1)
      ans_recall, ans_precision, key_match — answer-slot features: overlap of
                   the answer content (words not in the question) and whether
                   the key answer item is present (see answer_slot.py)

    Additional engineered features (12) are appended internally.  The
    XGBoost / MLP mixing weight is chosen on out-of-fold predictions of the
    training data, never on test data.

    The fitted model is saved as a plain .npz (XGBoost as JSON, the MLP and
    scaler as arrays), so loading never unpickles code.
    """

    BASE_FEATURES = ['base_sim', 'graph_sim', 'penalty', 'entity_mm', 'coverage', 'neg_mm', 'wrong_num',
                     'ans_recall', 'ans_precision', 'key_match']
    FEATURE_NAMES = BASE_FEATURES + [
        'sim*pen', 'sim*ent', 'grph*pen', 'pen*ent', 'sim_diff', 'avg_sim',
        'max_pen', 'cov*sim', 'neg*sim', 'num*sim', 'key*rec', 'rec*prec',
    ]
    WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)

    def __init__(self):
        self.xgb_model = self._new_xgb()
        self.mlp_model = self._new_mlp()
        self.scaler           = StandardScaler()
        self.is_trained       = False
        self.ensemble_weights = (0.6, 0.4)
        self._mlp_coefs = self._mlp_intercepts = None
        self._scale_mean = self._scale_std = None

    @staticmethod
    def _new_xgb():
        return xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE, verbosity=0)

    @staticmethod
    def _new_mlp():
        return MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', max_iter=600, early_stopping=True,
            validation_fraction=0.15, random_state=RANDOM_STATE,
            learning_rate='adaptive', learning_rate_init=0.001)

    # ─── Feature engineering ──────────────────────────────────────────────────

    def _engineer_features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(self.BASE_FEATURES):
            raise ValueError(f'Expected {len(self.BASE_FEATURES)} features '
                             f'{self.BASE_FEATURES}, got {X.shape[1]}')
        (base_sim, graph_sim, penalty, entity_mm, coverage, neg_mm, wrong_num,
         ans_recall, ans_precision, key_match) = (X[:, i:i + 1] for i in range(X.shape[1]))
        return np.hstack([
            X,
            base_sim  * penalty,    base_sim  * entity_mm,
            graph_sim * penalty,    penalty   * entity_mm,
            base_sim  - graph_sim,  (base_sim + graph_sim) / 2,
            np.maximum(penalty, entity_mm),
            coverage  * base_sim,   neg_mm * base_sim,
            wrong_num * base_sim,
            key_match * ans_recall, ans_recall * ans_precision,
        ])

    # ─── Training ─────────────────────────────────────────────────────────────

    def _choose_weight(self, X_eng, y, groups):
        """Pick the XGBoost weight minimising out-of-fold MAE on the training data."""
        n_splits = min(CV_N_SPLITS, len(y))
        if groups is not None and len(set(groups)) >= n_splits:
            folds = GroupKFold(n_splits=n_splits).split(X_eng, y, groups)
        else:
            folds = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE).split(X_eng)
        oof_x, oof_m = np.zeros(len(y)), np.zeros(len(y))
        for tr, va in folds:
            sc = StandardScaler().fit(X_eng[tr])
            oof_x[va] = self._new_xgb().fit(X_eng[tr], y[tr]).predict(X_eng[va])
            oof_m[va] = self._new_mlp().fit(sc.transform(X_eng[tr]), y[tr]).predict(sc.transform(X_eng[va]))
        maes = [np.mean(np.abs(np.clip(w * oof_x + (1 - w) * oof_m, 0, 100) - y))
                for w in self.WEIGHT_GRID]
        w = float(self.WEIGHT_GRID[int(np.argmin(maes))])
        print(f'Ensemble weight (out-of-fold, train only): XGBoost {w:.1f} / MLP {1 - w:.1f} '
              f'(OOF MAE {min(maes):.2f})')
        return (w, round(1 - w, 2))

    def train(self, X_train, y_train, groups=None):
        y = np.asarray(y_train, dtype=float)
        X_eng = self._engineer_features(X_train)
        self.ensemble_weights = self._choose_weight(X_eng, y, groups)

        X_s   = self.scaler.fit_transform(X_eng)
        self.xgb_model.fit(X_eng, y)
        self.mlp_model.fit(X_s, y)
        self._mlp_coefs      = [c.copy() for c in self.mlp_model.coefs_]
        self._mlp_intercepts = [b.copy() for b in self.mlp_model.intercepts_]
        self._scale_mean, self._scale_std = self.scaler.mean_.copy(), self.scaler.scale_.copy()
        self.is_trained = True

        imp = self.xgb_model.feature_importances_
        si  = np.argsort(imp)[::-1]
        print('Scorer trained. XGBoost importances (top 8):')
        for i in si[:8]:
            print(f'  {self.FEATURE_NAMES[i]:<16}: {imp[i]:.4f}')

    # ─── Inference ────────────────────────────────────────────────────────────

    def _mlp_predict(self, X_eng: np.ndarray) -> np.ndarray:
        """MLP forward pass from stored weights (ReLU hidden layers, identity output)."""
        a = (X_eng - self._scale_mean) / self._scale_std
        for i, (W, b) in enumerate(zip(self._mlp_coefs, self._mlp_intercepts)):
            a = a @ W + b
            if i < len(self._mlp_coefs) - 1:
                a = np.maximum(a, 0.0)
        return a.ravel()

    def predict_batch(self, X) -> np.ndarray:
        """Predict scores for a batch; returns array clipped to [0, 100]."""
        if not self.is_trained:
            raise ValueError('Train first.')
        X_eng = self._engineer_features(X)
        w1, w2 = self.ensemble_weights
        return np.clip(w1 * self.xgb_model.predict(X_eng) + w2 * self._mlp_predict(X_eng), 0, 100)

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save_model(self, filepath: str):
        """Save the ensemble to *filepath* (.npz, no pickle)."""
        if not self.is_trained:
            return
        raw = self.xgb_model.get_booster().save_raw(raw_format='json')
        arrays = {
            'xgb_json': np.frombuffer(bytes(raw), dtype=np.uint8),
            'ensemble_weights': np.array(self.ensemble_weights, dtype=float),
            'scale_mean': self._scale_mean, 'scale_std': self._scale_std,
            'n_layers': np.array(len(self._mlp_coefs)),
            'feature_names': np.array(self.FEATURE_NAMES),
        }
        for i, (W, b) in enumerate(zip(self._mlp_coefs, self._mlp_intercepts)):
            arrays[f'mlp_W{i}'], arrays[f'mlp_b{i}'] = W, b
        with open(filepath, 'wb') as f:
            np.savez(f, **arrays)
        print(f'Saved to {filepath}')

    def load_model(self, filepath: str):
        """Load a previously saved ensemble from *filepath*."""
        with np.load(filepath, allow_pickle=False) as d:
            if list(d['feature_names']) != self.FEATURE_NAMES:
                raise ValueError(f'{filepath} was saved with different features; retrain it.')
            self.xgb_model = xgb.XGBRegressor()
            self.xgb_model.load_model(bytearray(d['xgb_json'].tobytes()))
            self.ensemble_weights = tuple(float(w) for w in d['ensemble_weights'])
            self._scale_mean, self._scale_std = d['scale_mean'], d['scale_std']
            n = int(d['n_layers'])
            self._mlp_coefs      = [d[f'mlp_W{i}'] for i in range(n)]
            self._mlp_intercepts = [d[f'mlp_b{i}'] for i in range(n)]
        self.is_trained = True
        print(f'Loaded from {filepath}')
