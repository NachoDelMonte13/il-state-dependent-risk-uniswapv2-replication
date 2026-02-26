# ============================================================
# compute_shap_endogenous_scenarios_uniswapv2_pca3_xgboost.py
#
# Minimal replication code (Zenodo-ready)
# - Input:  <zenodo_dir>/<POOL>/panel_daily.csv
# - Output: <out_dir>/<POOL>/... (SHAP + PCA diagnostics)
#
# Run:
#   python code/compute_shap_endogenous_scenarios_uniswapv2_pca3_xgboost.py \
#       --pool WBTC_WETH --zenodo_dir ./data --out_dir ./results
# ============================================================

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


@dataclass(frozen=True)
class Config:
    target_future: str = "z_il_t_plus1"

    state_features: List[str] = (
        "z_il_t",
        "z_il_t_lag1",
        "z_price_implicit_lag1",
        "z_log_price",
        "z_delta_log_price_t_to_tplus1",  # shock feature (X_t)
    )

    z_tvl_col: str = "z_tvl_t"

    pca_components: int = 3
    test_size: float = 0.15
    shuffle: bool = False
    seed: int = 42

    shap_background_max: int = 500
    shap_nsamples: int = 200

    q_high_risk: float = 0.90
    q_liquidity_stress: float = 0.90

    xgb_params: Dict = None


CFG = Config(
    xgb_params=dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="reg:squarederror",
        random_state=42,
    )
)

SHOCK_FEATURE = "z_delta_log_price_t_to_tplus1"
STATE_ONLY_FEATURES = [f for f in CFG.state_features if f != SHOCK_FEATURE]


def select_background(X_train: pd.DataFrame, max_n: int, seed: int) -> pd.DataFrame:
    return X_train if len(X_train) <= max_n else X_train.sample(n=max_n, random_state=seed)


def mean_abs_shap(shap_values: np.ndarray, feature_names: List[str]) -> pd.Series:
    return pd.Series(
        np.nanmean(np.abs(shap_values), axis=0),
        index=feature_names
    ).sort_values(ascending=False)


def spearman_rank_corr(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    return np.nan if len(df) < 3 else df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank())


def aggregate_state_vs_shock_blocks(shap_values: np.ndarray, feature_names: List[str]) -> pd.Series:
    df = pd.DataFrame(np.abs(shap_values), columns=feature_names)
    shock = df[SHOCK_FEATURE].mean()
    state = df[STATE_ONLY_FEATURES].sum(axis=1).mean()
    total = shock + state
    return pd.Series({
        "shock_contribution": float(shock),
        "state_contribution": float(state),
        "shock_share": float(shock / total) if total != 0 else np.nan,
        "state_share": float(state / total) if total != 0 else np.nan,
    })


def compute_endogenous_liquidity_stress_abs_z_dlog_tvl(test_df: pd.DataFrame, z_tvl_col: str) -> pd.Series:
    d = test_df[z_tvl_col].astype(float).diff()
    z = (d - d.mean()) / d.std(ddof=0)
    return z.abs()


def define_scenarios(test_df: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    scenarios = {}

    thr = np.nanquantile(y_pred, CFG.q_high_risk)
    scenarios["high_predicted_risk"] = y_pred >= thr

    S_liq = compute_endogenous_liquidity_stress_abs_z_dlog_tvl(test_df, CFG.z_tvl_col)
    thr_liq = np.nanquantile(S_liq, CFG.q_liquidity_stress)
    scenarios["endogenous_liquidity_stress_abs_z_dlog_tvl"] = S_liq.values >= thr_liq

    return scenarios


def load_panel(panel_path: Path) -> pd.DataFrame:
    df = pd.read_csv(panel_path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if df["date"].isna().any():
        raise ValueError("Date parsing failed. Ensure 'date' is ISO (YYYY-MM-DD) in panel_daily.csv.")
    df = df.sort_values("date").reset_index(drop=True)

    required = ["date", CFG.target_future, CFG.z_tvl_col] + list(CFG.state_features)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in panel_daily.csv: {missing}")

    return df[required].dropna().reset_index(drop=True)


def run(pool: str, zenodo_dir: Path, out_dir: Path):
    panel_path = zenodo_dir / pool / "panel_daily.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"Expected input not found: {panel_path}")

    df = load_panel(panel_path)

    train_df, test_df = train_test_split(df, test_size=CFG.test_size, shuffle=False)

    X_train = train_df[list(CFG.state_features)]
    X_test = test_df[list(CFG.state_features)]
    y_train = train_df[CFG.target_future].values

    pca = PCA(n_components=CFG.pca_components, random_state=CFG.seed)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)

    model = XGBRegressor(**CFG.xgb_params)
    model.fit(Z_train, y_train)
    y_pred = model.predict(Z_test)

    def predict_from_raw(X):
        return model.predict(pca.transform(np.asarray(X)))

    explainer = shap.KernelExplainer(
        predict_from_raw,
        select_background(X_train, CFG.shap_background_max, CFG.seed).values
    )

    shap_values = np.asarray(
        explainer.shap_values(X_test.values, nsamples=CFG.shap_nsamples)
    )

    scenarios = define_scenarios(test_df, y_pred)

    out_pool = out_dir / pool
    out_pool.mkdir(parents=True, exist_ok=True)

    # --- GLOBAL ---
    global_mean = mean_abs_shap(shap_values, list(CFG.state_features))
    global_mean.to_csv(out_pool / "shap_global_mean_abs.csv")

    global_blocks = aggregate_state_vs_shock_blocks(shap_values, list(CFG.state_features))
    global_blocks.to_csv(out_pool / "shap_global_blocks_state_vs_shock.csv", header=False)

    # --- SCENARIOS ---
    summary_rows = []
    for name, mask in scenarios.items():
        scen_sv = shap_values[mask]
        if scen_sv.size == 0:
            pd.Series(dtype=float).to_csv(out_pool / f"shap_{name}_mean_abs.csv")
            pd.Series(dtype=float).to_csv(out_pool / f"shap_{name}_blocks_state_vs_shock.csv")
            summary_rows.append({
                "scenario": name,
                "n_obs": 0,
                "shock_share": np.nan,
                "state_share": np.nan,
                "spearman_rank_corr_vs_global": np.nan
            })
            continue

        scen_mean = mean_abs_shap(scen_sv, list(CFG.state_features))
        scen_mean.to_csv(out_pool / f"shap_{name}_mean_abs.csv")

        scen_blocks = aggregate_state_vs_shock_blocks(scen_sv, list(CFG.state_features))
        scen_blocks.to_csv(out_pool / f"shap_{name}_blocks_state_vs_shock.csv", header=False)

        spearman_rank = spearman_rank_corr(global_mean, scen_mean)

        summary_rows.append({
            "scenario": name,
            "n_obs": int(mask.sum()),
            "shock_share": float(scen_blocks.get("shock_share", np.nan)),
            "state_share": float(scen_blocks.get("state_share", np.nan)),
            "spearman_rank_corr_vs_global": float(spearman_rank) if spearman_rank == spearman_rank else np.nan
        })

    pd.DataFrame(summary_rows).to_csv(out_pool / "shap_scenarios_summary.csv", index=False)

    # --- PCA diagnostics ---
    loadings = pd.DataFrame(
        pca.components_.T,
        index=list(CFG.state_features),
        columns=[f"PC{i+1}" for i in range(CFG.pca_components)]
    )
    loadings.to_csv(out_pool / "pca_loadings_state_features.csv")

    explained = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(CFG.pca_components)],
        "explained_variance_ratio": pca.explained_variance_ratio_
    })
    explained.to_csv(out_pool / "pca_explained_variance.csv", index=False)

    print(f"[OK] Pool: {pool}")
    print(f"[OK] Input : {panel_path}")
    print(f"[OK] Output: {out_pool}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--zenodo_dir", required=True)
    parser.add_argument("--out_dir", default="./results")
    args = parser.parse_args()

    run(args.pool, Path(args.zenodo_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
