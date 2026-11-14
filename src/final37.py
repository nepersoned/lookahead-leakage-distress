"""모형 다양성 강건성 — 본문 설계(final27/final30)를 랜덤포레스트로 다시 잰다.

final30의 '로지스틱 회귀' 변형과 같은 방식으로, LightGBM 대신 다른 모형 계열을 쓴다.
    랜덤포레스트   sklearn RandomForestClassifier (나무 500개, 잎 최소 5, class_weight balanced_subsample,
                  중앙값 대치, 학습 구간 1·99% 분위수 절단)
핵심 비교 세 개: 누출(A − B), B+시장 − A, 남는 격차(A+시장 − B+시장).
출력: results/final37_score.csv, final37_ci.csv
실행: 저장소 루트에서 `python src/final37.py`
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import final7 as f7
import final27 as f27
import final28 as f28
import final29 as f29
import final30 as f30
import disclosure_mock as dm
import expand_analysis as ea
from dart import Dart

warnings.filterwarnings("ignore")
START, TEST = f30.START, f30.TEST


def rf_rolling(X, M, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    yr, yv = M.pred_year.values, M.y.values
    p = np.full(len(M), np.nan)
    for Y in TEST:
        tr, te = (yr >= START) & (yr < Y), yr == Y
        lo, hi = X[tr].quantile(.01), X[tr].quantile(.99)
        Xc = X.clip(lo, hi, axis=1).replace([np.inf, -np.inf], np.nan)
        keep = Xc.columns[Xc[tr].notna().any()]
        m = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                                 class_weight="balanced_subsample",
                                                 n_jobs=-1, random_state=seed))
        m.fit(Xc.loc[tr, keep], yv[tr])
        p[te] = m.predict_proba(Xc.loc[te, keep])[:, 1]
    return p[np.isin(yr, TEST)]


def main():
    rng = np.random.default_rng(42)
    _, pub, labels, mc, c2s = f7.load()
    ratios = pd.read_csv(f30.FULL / "classic_orig.csv", dtype={"corp_code": str})
    ratios["corp_code"] = ratios.corp_code.str.zfill(8)
    A, B, M = f7.design(ratios, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    A, B, M = A.reset_index(drop=True), B.reset_index(drop=True), M.reset_index(drop=True)

    CB = f27.disclosure(M)
    top = list(pd.read_csv("results/final27_shap.csv")["변수"][:10])
    sp = pd.read_csv("results/final27_sparse.csv")
    sel = top + [c for c in sp[sp.강제포함].변수 if c not in top]
    H = f28.history(M, labels)
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes.stock_code.astype(str).str.strip().str.zfill(6)
    c2s2 = (codes[codes.stock_code.str.len() == 6].drop_duplicates("corp_code")
            .set_index("corp_code")["stock_code"].to_dict())
    F = f29.market(M, c2s2)
    plus = lambda base: pd.concat([base, CB[sel], H, F], axis=1)

    te = M.pred_year.isin(TEST).values
    y = M.loc[te, "y"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)
    P = {"A": rf_rolling(A, M), "B": rf_rolling(B, M),
         "B+시장": rf_rolling(plus(B), M), "A+시장": rf_rolling(plus(A), M)}
    rows, cis = [], []
    for k, p in P.items():
        rows.append({"변형": "랜덤포레스트", "조건": k, "채점": int(te.sum()), "사건": int(y.sum()),
                     **{a: round(float(b), 4) for a, b in dm.score(y, p).items()}})
    for a, b, what in [("A", "B", "누출"), ("B+시장", "A", "정직 > 관행"), ("A+시장", "B+시장", "남는 격차")]:
        for r in ea.boot_ci(y, P[a], P[b], grp, 1000, rng):
            f_ = r["0초과"]
            cis.append({"변형": "랜덤포레스트", "비교": what, **r, "p(양측)": round(max(2 * min(f_, 1 - f_), 0.002), 3)})
    pd.DataFrame(rows).to_csv("results/final37_score.csv", index=False, encoding="utf-8-sig")
    ci = pd.DataFrame(cis)
    ci.to_csv("results/final37_ci.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(), flush=True)
    print(ci[ci.지표 == "PR-AUC"].to_string(), flush=True)


if __name__ == "__main__":
    main()
