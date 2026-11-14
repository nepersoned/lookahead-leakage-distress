# data/

Raw and intermediate data are not redistributed here. All inputs come from public sources and are rebuilt by the collection scripts in `src/`:

| File(s) | Source | Script |
|---|---|---|
| `full/reports_full*.csv` (periodic-report filing dates) | DART Open API | `gate3a_reports.py` |
| `full/kind_events*.csv`, `raw/kind_manual/` (designations, releases, delistings) | KRX KIND | `kind.py`, `kind_codes.py` |
| `full/orig_fin2*.csv`, `full/classic_orig*.csv` (originally filed statements) | DART original documents | `orig_fin.py`, `orig_fin2.py`, `vars_classic.py` |
| `full/classic_full.csv` (current, amended values) | DART Open API | `vars21.py`, `vars_classic.py` |
| `full/q3_fin.csv` (Q3 reports) | DART Open API | `q3_fin.py` |
| `full/filings_all.csv` (non-periodic disclosures) | DART Open API | `filings_all.py` |
| `full/auditors_full.csv` | DART Open API | `auditors.py` |
| `full/marketcap.csv`, `market_own/release/` (daily trading) | KRX Open API | `marketcap.py`, `market_collect.py` |

A DART API key is required (`.env`, see `.env.example`).
