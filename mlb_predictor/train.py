"""Trains the run model: two XGBoost Poisson regressors (home runs, away
runs) over the 20-feature frame, evaluated on a time-based split."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from xgboost import XGBRegressor

from .config import FEATURE_COLS, MODELS_DIR
from .features import outcome_labels, training_frame
from .predict import win_probabilities

MODEL_PATH = MODELS_DIR / "mlb_xgb.joblib"
REPORT_PATH = MODELS_DIR / "mlb_eval_report.json"

# Recent games count more. A game this many days old is worth half as much
# as today's in training. Baseball rosters and form shift within a season
# (trades, slumps, call-ups), so a ~1-season half-life keeps it current
# without throwing away the multi-year sample.
FORM_HALF_LIFE_DAYS = 365

XGB_PARAMS = dict(
    objective="count:poisson",
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=3.0,
    random_state=26,
)


def _time_decay_weights(dates: pd.Series) -> np.ndarray:
    """Exponential recency weights: newest game = 1.0, halving every
    FORM_HALF_LIFE_DAYS."""
    age_days = (dates.max() - dates).dt.days.to_numpy()
    return np.power(0.5, age_days / FORM_HALF_LIFE_DAYS)


def _fit(X, y, weights=None) -> XGBRegressor:
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X, y, sample_weight=weights)
    return model


def main(test_season: int = 2026) -> None:
    df = training_frame()
    train = df[df["season"] < test_season]
    test = df[df["season"] >= test_season]
    print(f"Training on {len(train)} games (< {test_season}), "
          f"evaluating on {len(test)} played {test_season} games")

    X_train, X_test = train[FEATURE_COLS], test[FEATURE_COLS]
    w_train = _time_decay_weights(train["date"])
    home_model = _fit(X_train, train["home_runs"], w_train)
    away_model = _fit(X_train, train["away_runs"], w_train)

    report = {"train_games": int(len(train)), "test_games": int(len(test)),
              "test_season": test_season}
    if len(test):
        mu_home = home_model.predict(X_test)
        mu_away = away_model.predict(X_test)
        probs = np.array([win_probabilities(h, a) for h, a in zip(mu_home, mu_away)])
        p_home = probs[:, 0]
        y_true = outcome_labels(test)  # 1 = home win
        base_rate = float(y_true.mean())

        report.update({
            "mae_home_runs": round(float(mean_absolute_error(test["home_runs"], mu_home)), 3),
            "mae_away_runs": round(float(mean_absolute_error(test["away_runs"], mu_away)), 3),
            "home_win_accuracy": round(float(accuracy_score(y_true, (p_home >= 0.5).astype(int))), 3),
            "log_loss": round(float(log_loss(y_true, p_home, labels=[0, 1])), 3),
            "baseline_accuracy_always_home": round(base_rate, 3),
            "baseline_log_loss": round(float(
                log_loss(y_true, np.full_like(p_home, base_rate), labels=[0, 1])), 3),
        })

    X_all = df[FEATURE_COLS]
    w_all = _time_decay_weights(df["date"])
    bundle = {
        "home_model": _fit(X_all, df["home_runs"], w_all),
        "away_model": _fit(X_all, df["away_runs"], w_all),
        "feature_cols": FEATURE_COLS,
        "trained_on_games": int(len(df)),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"Saved model -> {MODEL_PATH}")


def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model — run `python -m mlb_predictor train` first.")
    return joblib.load(MODEL_PATH)
