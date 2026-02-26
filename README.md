# Replication Package: State-Dependent Risk in DeFi Markets (Uniswap v2 Impermanent Loss)

This repository provides an analysis-ready replication package for the manuscript:

State-Dependent Risk in DeFi Markets: Evidence from Impermanent Loss in Constant-Product AMMs  
Ignacio Ariel Del Monte; Juan José De Lucio; Miguel Angel Sicilia; Sebastian Heil

The package includes (i) daily pool-level panels used in the empirical analysis and (ii) minimal replication code to reproduce the SHAP-based explainability exercise under endogenous scenario definitions.

Contact (data and code): Ignacio Ariel Del Monte - ignacio.monte@uah.es 

## Contents

Repository structure:

.
├─ code/
│  └─ compute_shap_endogenous_scenarios_uniswapv2_pca3_xgboost.py
├─ data/
│  ├─ WBTC_WETH/
│  │  └─ panel_daily.csv
│  ├─ DAI_WETH/
│  │  └─ panel_daily.csv
│  └─ (additional pools, if provided)
├─ requirements.txt
├─ LICENSE_CODE.txt
├─ LICENSE_DATA.txt
└─ CITATION.cff

## Data overview

Each pool folder under data/<POOL>/ contains a daily panel file:

data/<POOL>/panel_daily.csv

The panel is indexed at the pool-day level and aligned with the manuscript’s empirical design:
- sample period: 2021-06-01 to 2024-05-31
- chronological split: 85% training / 15% test
- pool-level standardization (z-scores)

This repository is intended to replicate the manuscript results using the analysis-ready dataset. It does not reproduce the full upstream extraction and cleaning pipeline from Dune.

## Required columns

The replication script expects the following columns in data/<POOL>/panel_daily.csv:

- date (ISO format: YYYY-MM-DD)
- z_il_t_plus1
- z_tvl_t
- z_il_t
- z_il_t_lag1
- z_price_implicit_lag1
- z_log_price
- z_delta_log_price_t_to_tplus1

The file may include additional columns; they are ignored by the script.

## Replication code

Script:

code/compute_shap_endogenous_scenarios_uniswapv2_pca3_xgboost.py

The script implements:
- chronological split (85/15)
- PCA with 3 components, estimated on the training sample only and applied to the test sample
- XGBoost regression model trained on PCA scores
- SHAP KernelExplainer computed on the test sample
- two ex-post test-only scenarios:
  1) high predicted risk (top 10% of predicted values in the test set)
  2) endogenous liquidity stress based on |z(Δ log TVL)| in the test set (top 10%)

## Installation

Python 3.10+ is recommended.

Install dependencies:

pip install -r requirements.txt

## Running the replication

From the repository root:

python code/compute_shap_endogenous_scenarios_uniswapv2_pca3_xgboost.py --pool WBTC_WETH --zenodo_dir data --out_dir results

Replace WBTC_WETH with any pool folder name available under data/.

## Outputs

For each pool, the script creates:

results/<POOL>/
- shap_global_mean_abs.csv
- shap_global_blocks_state_vs_shock.csv
- shap_high_predicted_risk_mean_abs.csv
- shap_high_predicted_risk_blocks_state_vs_shock.csv
- shap_endogenous_liquidity_stress_abs_z_dlog_tvl_mean_abs.csv
- shap_endogenous_liquidity_stress_abs_z_dlog_tvl_blocks_state_vs_shock.csv
- shap_scenarios_summary.csv
- pca_loadings_state_features.csv
- pca_explained_variance.csv

## Reference outputs (quick verification)

The folder results_example/ contains reference outputs generated with the replication script for the benchmark pool(s). These files are provided to enable quick verification without executing the code.

If you run the script, you should reproduce the same set of output files. Minor numerical differences may occur across platforms due to library versions and floating-point arithmetic.

## Citation

If you use the data or code, please cite the manuscript and this repository (Zenodo DOI):

Zenodo DOI: to be added after the first Zenodo release

A machine-readable citation entry is provided in CITATION.cff.

## License

This repository includes separate licenses for code and data:
- Code: see LICENSE_CODE.txt
- Data: see LICENSE_DATA.txt

The data license restricts use to non-commercial purposes and requires attribution.
