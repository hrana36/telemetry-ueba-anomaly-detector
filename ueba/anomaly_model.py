"""
Dual-Ensemble Unsupervised UEBA Anomaly Detection Engine
Combines Isolation Forest sub-space partitioning with PCA Reconstruction Error.
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DualEnsembleUEBAModel:
    """
    Hybrid Unsupervised Anomaly Detection Model combining:
    1. Isolation Forest (Tree-based partitioning of point anomalies)
    2. Principal Component Analysis (Reconstruction error of multivariate subspace violations)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        pca_variance_ratio: float = 0.90,
        alpha_ensemble_weight: float = 0.5,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.pca_variance_ratio = pca_variance_ratio
        self.alpha = alpha_ensemble_weight
        self.contamination = contamination
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.pca: Optional[PCA] = None
        self.is_fitted = False

        # Baseline statistics for normalization and explanation
        self.mean_vector: Optional[np.ndarray] = None
        self.std_vector: Optional[np.ndarray] = None
        self.if_min = 0.0
        self.if_max = 1.0
        self.pca_min = 0.0
        self.pca_max = 1.0

    def fit(self, X: np.ndarray) -> "DualEnsembleUEBAModel":
        """
        Fits baseline behavioral distributions on training telemetry feature matrix.
        X shape: (n_samples, n_features)
        """
        if len(X) < 3:
            raise ValueError("Training matrix requires at least 3 samples to fit PCA and Isolation Forest.")

        X_scaled = self.scaler.fit_transform(X)
        self.mean_vector = np.mean(X, axis=0)
        self.std_vector = np.maximum(np.std(X, axis=0), 0.05)

        # Fit Isolation Forest
        self.iso_forest.fit(X_scaled)
        # raw decision function: lower means more abnormal
        raw_if = -self.iso_forest.decision_function(X_scaled)
        self.if_mean = float(np.mean(raw_if))
        self.if_std = max(float(np.std(raw_if)), 0.02)

        # Fit PCA
        max_components = min(X_scaled.shape[0] - 1, X_scaled.shape[1] - 1, 4)
        n_comp = max(1, max_components)
        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        self.pca.fit(X_scaled)

        # PCA Reconstruction Error on training set
        X_proj = self.pca.inverse_transform(self.pca.transform(X_scaled))
        recon_errors = np.mean(np.square(X_scaled - X_proj), axis=1)
        self.pca_mean = float(np.mean(recon_errors))
        self.pca_std = max(float(np.std(recon_errors)), 0.05)

        self.is_fitted = True
        return self

    def _calibrate_score(self, raw_score: float, mean: float, std: float, offset: float = 2.0) -> float:
        """
        Calibrates raw divergence scores via extreme-value logistic sigmoid:
        P = 1 / (1 + exp(-(z - offset)))
        """
        z = (raw_score - mean) / std
        prob = 1.0 / (1.0 + np.exp(-1.0 * (z - offset)))
        return float(np.clip(prob, 0.0, 1.0))

    def predict_score(self, x_vec: np.ndarray) -> Tuple[float, float, float]:
        """
        Evaluates a single feature vector.
        Returns:
            (composite_score, norm_isolation_score, norm_pca_error)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")

        x_2d = x_vec.reshape(1, -1)
        x_scaled = self.scaler.transform(x_2d)

        # 1. Isolation Forest score
        raw_if = float(-self.iso_forest.decision_function(x_scaled)[0])
        norm_if = self._calibrate_score(raw_if, self.if_mean, self.if_std)

        # 2. PCA Reconstruction Error
        x_proj = self.pca.inverse_transform(self.pca.transform(x_scaled))
        recon_error = float(np.mean(np.square(x_scaled - x_proj)))
        norm_pca = self._calibrate_score(recon_error, self.pca_mean, self.pca_std)

        # Composite fused score
        composite = self.alpha * norm_if + (1.0 - self.alpha) * norm_pca
        return round(composite, 4), round(norm_if, 4), round(norm_pca, 4)

    def explain_vector(self, x_vec: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        """
        Computes normalized z-score deviation and reconstruction residual attribution per feature.
        """
        if not self.is_fitted:
            return []

        x_2d = x_vec.reshape(1, -1)
        x_scaled = self.scaler.transform(x_2d)
        x_proj = self.pca.inverse_transform(self.pca.transform(x_scaled))
        residuals = np.abs(x_scaled - x_proj).flatten()

        z_scores = np.abs((x_vec - self.mean_vector) / self.std_vector)

        attributions = []
        for i, name in enumerate(feature_names):
            score = float(0.5 * z_scores[i] + 0.5 * residuals[i])
            attributions.append({
                "feature": name,
                "observed_value": round(float(x_vec[i]), 4),
                "baseline_mean": round(float(self.mean_vector[i]), 4),
                "z_score": round(float(z_scores[i]), 2),
                "anomaly_contribution": round(score, 4),
            })

        # Rank by contribution descending
        attributions.sort(key=lambda d: d["anomaly_contribution"], reverse=True)
        return attributions
