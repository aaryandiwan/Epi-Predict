"""
Epi Predict – Model Explainability Module

Provides SHAP-based explanations for tree-ensemble models (XGBoost,
Random Forest) and generic feature-importance extraction for any
scikit-learn-compatible estimator.

Key capabilities:
    • SHAP value computation via ``TreeExplainer``
    • Ranked feature importance extraction
    • Human-readable natural-language explanations
    • Cross-model feature importance comparison

The module degrades gracefully when:
    - The ``shap`` package is not installed
    - A model does not expose ``feature_importances_``
    - A saved model file is missing or corrupt

Author : Epi Predict Team
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from config.settings import MODELS_DIR, MODEL_REGISTRY_FILE, FEATURE_COLUMNS

logger = logging.getLogger("epi_predict.explainability")

# ─── SHAP availability check ────────────────────────────────────────────────
try:
    import shap

    _SHAP_AVAILABLE = True
    logger.debug("SHAP library detected (version %s).", shap.__version__)
except ImportError:
    _SHAP_AVAILABLE = False
    logger.info(
        "SHAP library not installed. SHAP-based explanations will be unavailable. "
        "Install with: pip install shap"
    )

# Models that support TreeExplainer
_TREE_MODELS = {"xgboost", "random_forest", "tuned_random_forest"}


# ─── Helper: Load a model from disk ─────────────────────────────────────────

def _load_model(model_name: str):
    """Load a trained model from the model registry.

    Args:
        model_name: Key in the model registry (e.g. ``"xgboost"``).

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If registry or model artefact is missing.
        ValueError: If *model_name* is not in the registry.
    """
    import joblib

    if not MODEL_REGISTRY_FILE.exists():
        raise FileNotFoundError(
            f"Model registry not found at {MODEL_REGISTRY_FILE}. "
            "Train models first."
        )

    with open(MODEL_REGISTRY_FILE, "r") as fh:
        registry: dict = json.load(fh)

    if model_name not in registry:
        available = [k for k in registry if not k.startswith("_")]
        raise ValueError(
            f"Model '{model_name}' not in registry. Available: {available}"
        )

    model_path = Path(registry[model_name]["path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = joblib.load(model_path)
    logger.info("Loaded model '%s' from %s", model_name, model_path)
    return model


def _get_feature_names(model_name: str) -> List[str]:
    """Return the feature names used during training.

    Checks the model registry for stored feature names; falls back to
    ``FEATURE_COLUMNS`` from settings.
    """
    try:
        with open(MODEL_REGISTRY_FILE, "r") as fh:
            registry = json.load(fh)
        features = registry.get(model_name, {}).get("features")
        if features:
            return features
    except Exception:
        pass
    return list(FEATURE_COLUMNS)


# ─── SHAP Analysis ──────────────────────────────────────────────────────────

def get_shap_analysis(
    model_name: str,
    X_sample: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute SHAP values for a model using ``TreeExplainer``.

    Args:
        model_name: Registry key of the model to explain.
        X_sample: Optional feature matrix (n_samples × n_features).
            If ``None``, a small synthetic sample is generated for
            demonstration purposes.

    Returns:
        Dictionary with:
            - **model_name** (str): Echo of the input model name.
            - **shap_available** (bool): Whether SHAP was usable.
            - **shap_values** (list[list[float]] | None): Per-sample,
              per-feature SHAP values.
            - **base_value** (float | None): Expected value (base rate).
            - **feature_names** (list[str]): Feature column names.
            - **mean_abs_shap** (list[dict] | None): Ranked mean |SHAP|
              per feature.
            - **message** (str): Status / error message.

    Example::

        >>> analysis = get_shap_analysis("xgboost", X_test[:50])
        >>> analysis["mean_abs_shap"][0]
        {'feature': 'lag_1', 'importance': 0.412}
    """
    result: Dict[str, Any] = {
        "model_name": model_name,
        "shap_available": False,
        "shap_values": None,
        "base_value": None,
        "feature_names": [],
        "mean_abs_shap": None,
        "message": "",
    }

    # Gate: SHAP installed?
    if not _SHAP_AVAILABLE:
        result["message"] = (
            "SHAP library is not installed. "
            "Install with: pip install shap"
        )
        logger.warning(result["message"])
        return result

    # Gate: tree model?
    if model_name not in _TREE_MODELS:
        result["message"] = (
            f"SHAP TreeExplainer is only supported for tree-based models "
            f"({', '.join(sorted(_TREE_MODELS))}). "
            f"Model '{model_name}' is not supported."
        )
        logger.info(result["message"])
        return result

    try:
        model = _load_model(model_name)
    except (FileNotFoundError, ValueError) as exc:
        result["message"] = str(exc)
        logger.error("Cannot load model for SHAP: %s", exc)
        return result

    feature_names = _get_feature_names(model_name)
    result["feature_names"] = feature_names

    # Fallback sample
    if X_sample is None:
        logger.info(
            "No X_sample provided; generating synthetic sample with %d features.",
            len(feature_names),
        )
        rng = np.random.default_rng(42)
        X_sample = rng.standard_normal((10, len(feature_names)))

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # TreeExplainer returns ndarray for regression
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        base_value = float(explainer.expected_value)
        if isinstance(explainer.expected_value, np.ndarray):
            base_value = float(explainer.expected_value[0])

        # Mean absolute SHAP per feature (global importance)
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        ranked_indices = np.argsort(mean_abs)[::-1]
        mean_abs_shap = [
            {
                "feature": (
                    feature_names[i] if i < len(feature_names) else f"feature_{i}"
                ),
                "importance": round(float(mean_abs[i]), 6),
            }
            for i in ranked_indices
        ]

        result.update(
            {
                "shap_available": True,
                "shap_values": shap_values.tolist(),
                "base_value": round(base_value, 6),
                "mean_abs_shap": mean_abs_shap,
                "message": "SHAP analysis completed successfully.",
            }
        )

        logger.info(
            "SHAP analysis for '%s': %d samples, top feature = %s (%.4f)",
            model_name,
            len(X_sample),
            mean_abs_shap[0]["feature"],
            mean_abs_shap[0]["importance"],
        )

    except Exception as exc:
        result["message"] = f"SHAP computation failed: {exc}"
        logger.error(result["message"], exc_info=True)

    return result


# ─── Feature Importance ──────────────────────────────────────────────────────

def get_feature_importance(model_name: str) -> List[Dict[str, Any]]:
    """Extract and rank feature importance from a trained model.

    Works with any estimator that exposes ``feature_importances_``
    (tree ensembles) or ``coef_`` (linear models).

    Args:
        model_name: Registry key of the model.

    Returns:
        List of dicts sorted by descending importance::

            [{"feature": "lag_1", "importance": 0.42, "rank": 1}, ...]

        Returns an empty list if importance extraction fails.
    """
    try:
        model = _load_model(model_name)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Cannot load model for importance: %s", exc)
        return []

    feature_names = _get_feature_names(model_name)
    importances: Optional[np.ndarray] = None

    # Tree-based models
    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_)
        logger.debug("Using feature_importances_ from %s", model_name)

    # Linear models (use absolute coefficient values)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.array(model.coef_).ravel())
        logger.debug("Using |coef_| from %s", model_name)

    # Stacking – try the final estimator
    elif hasattr(model, "final_estimator_"):
        if hasattr(model.final_estimator_, "coef_"):
            importances = np.abs(np.array(model.final_estimator_.coef_).ravel())
            logger.debug("Using final_estimator_.coef_ from stacking model")

    if importances is None:
        logger.warning(
            "Model '%s' does not expose feature_importances_ or coef_.",
            model_name,
        )
        return []

    # Normalise to sum to 1
    total = importances.sum()
    if total > 0:
        importances = importances / total

    # Align with feature names
    n = min(len(importances), len(feature_names))
    ranked_indices = np.argsort(importances[:n])[::-1]

    result = [
        {
            "feature": feature_names[i],
            "importance": round(float(importances[i]), 6),
            "rank": rank + 1,
        }
        for rank, i in enumerate(ranked_indices)
    ]

    logger.info(
        "Feature importance for '%s': top=%s (%.4f)",
        model_name,
        result[0]["feature"] if result else "N/A",
        result[0]["importance"] if result else 0.0,
    )

    return result


# ─── Human-Readable Explanation ──────────────────────────────────────────────

def generate_explanation_text(
    prediction: float,
    shap_values: Optional[List[float]],
    features: List[str],
) -> str:
    """Generate a natural-language explanation for a single prediction.

    Combines the predicted value with SHAP attributions to produce
    a paragraph suitable for display in a dashboard tooltip or report.

    Args:
        prediction: The model's predicted case count.
        shap_values: SHAP values for this single prediction (one per
            feature). May be ``None`` if SHAP is unavailable.
        features: Feature names corresponding to *shap_values*.

    Returns:
        Multi-sentence explanation string.

    Example::

        >>> text = generate_explanation_text(1500.0, [200, -50, 800, ...], [...])
        >>> print(text)
        The model predicts approximately 1,500 influenza cases. ...
    """
    pred_str = f"{prediction:,.0f}"
    lines = [f"The model predicts approximately {pred_str} influenza cases."]

    if shap_values is None or not features:
        lines.append(
            "Detailed feature-level attribution is not available for this "
            "prediction (SHAP values were not computed)."
        )
        return " ".join(lines)

    # Pair features with SHAP values and sort by |impact|
    pairs = list(zip(features, shap_values))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)

    # Top 3 contributors
    top_n = min(3, len(pairs))
    drivers = []
    for feat, val in pairs[:top_n]:
        direction = "increasing" if val > 0 else "decreasing"
        drivers.append(
            f"'{feat}' ({direction} the prediction by ~{abs(val):,.0f})"
        )

    lines.append(
        f"The top contributing factors are: {'; '.join(drivers)}."
    )

    # Count positive vs negative contributors
    pos = sum(1 for _, v in pairs if v > 0)
    neg = sum(1 for _, v in pairs if v < 0)
    lines.append(
        f"Overall, {pos} feature(s) push the prediction higher while "
        f"{neg} feature(s) push it lower."
    )

    explanation = " ".join(lines)
    logger.debug("Generated explanation (%d chars).", len(explanation))
    return explanation


# ─── Cross-Model Comparison ─────────────────────────────────────────────────

def compare_model_explanations() -> Dict[str, Any]:
    """Compare feature importance across all available models.

    Loads each model from the registry, extracts feature importance,
    and assembles a side-by-side comparison matrix.

    Returns:
        Dictionary with:
            - **models** (dict): Keyed by model name, each containing
              the importance list from :func:`get_feature_importance`.
            - **common_top_features** (list[str]): Features appearing
              in the top-3 of *every* model that reported importance.
            - **num_models_compared** (int): Count of models analysed.
            - **message** (str): Status message.
    """
    result: Dict[str, Any] = {
        "models": {},
        "common_top_features": [],
        "num_models_compared": 0,
        "message": "",
    }

    # Load registry
    if not MODEL_REGISTRY_FILE.exists():
        result["message"] = "Model registry not found. Train models first."
        logger.warning(result["message"])
        return result

    with open(MODEL_REGISTRY_FILE, "r") as fh:
        registry = json.load(fh)

    model_names = [k for k in registry if not k.startswith("_")]
    top3_sets: List[set] = []

    for name in model_names:
        try:
            importance = get_feature_importance(name)
            if importance:
                result["models"][name] = importance
                result["num_models_compared"] += 1
                top3 = {item["feature"] for item in importance[:3]}
                top3_sets.append(top3)
        except Exception as exc:
            logger.warning("Skipping model '%s' in comparison: %s", name, exc)
            result["models"][name] = {"error": str(exc)}

    # Intersection of all top-3 sets
    if top3_sets:
        common = top3_sets[0]
        for s in top3_sets[1:]:
            common = common & s
        result["common_top_features"] = sorted(common)

    result["message"] = (
        f"Compared {result['num_models_compared']} model(s). "
        f"Common top features: {result['common_top_features'] or 'none'}."
    )

    logger.info(result["message"])
    return result
