"""
scorer.py - XGBoost + MLP Ensemble Answer Scorer with feature engineering.
"""

import numpy as np
import joblib
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.config import SCORER_MODEL_PATH, ENSEMBLE_WEIGHTS

class AnswerScorer:
    """Ensemble scorer combining XGBoost + MLP regressor."""

    _FEAT_NAMES = [
        "base_sim", "graph_sim", "penalty", "entity_mm",
        "sim×penalty", "sim×entity", "graph×penalty",
        "penalty×entity", "sim_diff", "avg_sim", "max_penalty",
    ]

    def __init__(self):
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0,
        )
        self.mlp_model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16), activation="relu", solver="adam",
            max_iter=500, early_stopping=True, validation_fraction=0.15,
            random_state=42, learning_rate="adaptive", learning_rate_init=0.001,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.ensemble_weights = ENSEMBLE_WEIGHTS

    # ── feature engineering ───────────────────────────────────────────────────

    def _engineer_features(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        bs, gs, pen, emm = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]

        return np.hstack([
            X,
            bs * pen,
            bs * emm,
            gs * pen,
            pen * emm,
            bs - gs,
            (bs + gs) / 2,
            np.maximum(pen, emm),
        ])

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, X_train, y_train) -> None:
        X_eng = self._engineer_features(X_train)
        X_scaled = self.scaler.fit_transform(X_eng)

        self.xgb_model.fit(X_eng, y_train)
        self.mlp_model.fit(X_scaled, y_train)
        self.is_trained = True
        importances = self.xgb_model.feature_importances_
        order = np.argsort(importances)[::-1]
        print("Ensemble Scorer trained. XGBoost Feature Importances:")
        for i in order[:6]:
            print(f"  {self._FEAT_NAMES[i]:<16}: {importances[i]:.4f}")

    # ── inference ─────────────────────────────────────────────────────────────

    def predict_score(self, baseline_sim: float, graph_sim: float,
                      penalty: float, entity_mismatch: float = 0.0) -> float:
        """Return a score in [0, 100] for a single sample."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        features = np.array([[baseline_sim, graph_sim, penalty, entity_mismatch]])
        X_eng = self._engineer_features(features)
        X_scaled = self.scaler.transform(X_eng)

        w_xgb, w_mlp = self.ensemble_weights
        raw = w_xgb * self.xgb_model.predict(X_eng)[0] + w_mlp * self.mlp_model.predict(X_scaled)[0]
        return float(np.clip(raw, 0.0, 100.0))

    def predict_batch(self, X) -> np.ndarray:
        """Batch prediction; returns array of scores in [0, 100]."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        X_eng = self._engineer_features(X)
        X_scaled = self.scaler.transform(X_eng)
        w_xgb, w_mlp = self.ensemble_weights
        return np.clip(
            w_xgb * self.xgb_model.predict(X_eng) + w_mlp * self.mlp_model.predict(X_scaled),
            0, 100,
        )

    # ── explanation ───────────────────────────────────────────────────────────

    def generate_explanation(self, score: float, baseline_sim: float, graph_sim: float,
                              penalty: float, explanations: list,
                              entity_mismatch: float = 0.0) -> str:
        lines = [
            "=" * 40,
            f"  Final Score: {score:.2f} / 100",
            "=" * 40,
            f"  Semantic Similarity (LaBSE):     {baseline_sim:.4f}",
            f"  Structural Similarity (GAT-GNN): {graph_sim:.4f}",
            f"  Entity Mismatch:                 {entity_mismatch:.4f}",
        ]
        if penalty > 0:
            lines.append(f"\n  ⚠ Role Errors Detected (Penalty: {penalty:.2f}):")
            for exp in explanations:
                lines.append(f"    - {exp}")
        else:
            lines.append("\n  ✓ No semantic role errors detected.")
        lines.append("=" * 40)
        return "\n".join(lines)

    # ── persistence ───────────────────────────────────────────────────────────

    def save_model(self, filepath: str = SCORER_MODEL_PATH) -> None:
        if self.is_trained:
            joblib.dump(
                {
                    "xgb_model": self.xgb_model,
                    "mlp_model": self.mlp_model,
                    "scaler": self.scaler,
                    "ensemble_weights": self.ensemble_weights,
                },
                filepath,
            )
            print(f"Ensemble scorer saved to {filepath}")

    def load_model(self, filepath: str = SCORER_MODEL_PATH) -> None:
        data = joblib.load(filepath)
        self.xgb_model = data["xgb_model"]
        self.mlp_model = data["mlp_model"]
        self.scaler = data["scaler"]
        self.ensemble_weights = data["ensemble_weights"]
        self.is_trained = True
        print(f"Ensemble scorer loaded from {filepath}")
