# Look-ahead leakage and data vintages in distress early-warning models

Code, results and pre-registrations for the paper on look-ahead leakage in KOSDAQ administrative-issue designation prediction (manuscript in preparation).

## What the code does
On the same data, model and sample, the analyses compare
- **A** — financial statements attached by fiscal year-end (common practice; at a January 1 prediction date the FY(Y−1) annual report is not yet public),
- **B** — statements attached by actual DART filing date (only what was public at the prediction date),
- **B′** — B plus the prior-year Q3 report, already public on January 1,
- **A (amended)** — A using current, later-amended values from the DART API instead of originally filed figures,

and measure the performance difference (leakage), when it arises, which variables carry it, and how much is due to later amendments (data vintage).

## Layout
| Path | Contents |
|---|---|
| `src/` | collection (`dart.py`, `kind.py`, `orig_fin2.py`, `q3_fin.py`, `market_collect.py`, …) and analysis scripts (`final27.py`–`final52.py`) |
| `src/run_all.sh` | runs the analyses in order after data collection |
| `prereg/` | pre-specified plans committed before scoring: 2025 holdout, variable-swap mechanism test |
| `results/` | output tables (CSV/log) and the literature audit (`lit_audit_timing.csv`) |
| `data/` | not redistributed; see `data/README.md` |

## Main scripts
| Script | Analysis |
|---|---|
| `final27.py` | main comparison (A, B, B+, A+, B+Market, A+Market), 1:1 replication, selection |
| `final30.py`, `final37.py`, `final44.py` | design variations, random forest, Shumway-type hazard and accounting logit |
| `final31.py`, `final34.py`, `final45.py` | fixed-lag rules, prediction-date shifts, April anomaly |
| `final32.py`, `final46.py`, `final48.py` | delisting, event types, event types against B′ |
| `final38.py`, `final39.py` | 2025 holdout (pre-specified), 2019–2025 pooled |
| `final40.py` | recalibrated Brier/log scores, calibration, top-k CIs, Holm adjustment |
| `final41.py`, `final50.py` | vintage decomposition (filing-lag vs. amendment), amendment audit |
| `final42.py`, `final43.py` | point-in-time population, training-perturbation sensitivity |
| `final47.py` | B′ (Q3 reports) |
| `final49.py` | pre-registered variable-swap mechanism test |
| `final51.py`, `final52.py` | composite scores removed; K-Score with original K2 coefficients |

The K-Score uses the original K2 coefficients of Altman, Eom & Kim (1994, NYU Salomon Center WP FIN-94-002, p. 9).

## Notes
- Requires Python 3.11, pandas, scikit-learn, lightgbm, shap, and a DART Open API key in `.env`.
- Generative AI (Claude, Gemini) assisted with code and writing; all outputs were checked by the author.
