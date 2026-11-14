"""최종 4조건 — A · A+ · B · B+ (코스닥, 평가창 A·B).

기준선(고정): K-Score 4변수+점수, Ohlson O 9변수+점수. 원식 FUTL(세전이익+감가상각비)은
72.8% 결측이라 영업현금흐름/총부채로 대체하고 O_SCORE도 그 값으로 다시 계산한다.

공시 변수: 후보 전부를 B에 붙여 예측연도 2013~2017(두 평가창 공통 학습 구간)에서만
LightGBM을 학습하고, 학습 데이터의 평균 |SHAP| 상위 10개를 고른다.
그 한 목록을 A+·B+, 평가창 A·B 모두에 똑같이 쓴다.

출력: results/final4_select.csv, final4_score.csv, final4_ci.csv, final4_formula.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import disclosure_mock as dm
import expand_analysis as ea
import feature_screen as fs
import gate3e_full as g

warnings.filterwarnings("ignore")
K = 10
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1000
BASE = ["K_lnTA", "K_lnSLTA", "K_RETA", "K_METL", "KSCORE",
        "O_SIZE", "O_TLTA", "O_WCTA", "O_CLCA", "O_OENEG", "O_NITA", "O_FUTL_CFO",
        "O_INTWO", "O_CHIN", "O_SCORE"]


def main():
    rng = np.random.default_rng(42)
    ratios, pub, labels = ea.load_market("KOSDAQ", "classic")
    o = ratios
    ratios["O_SCORE"] = (-1.32 - 0.407 * o.O_SIZE + 6.03 * o.O_TLTA - 1.43 * o.O_WCTA
                         + 0.0757 * o.O_CLCA - 1.72 * o.O_OENEG - 2.37 * o.O_NITA
                         - 1.83 * o.O_FUTL_CFO + 0.285 * o.O_INTWO - 0.521 * o.O_CHIN)
    g.RATIO_COLS = BASE
    years = list(range(2013, 2025))
    A, B, M = g.build(ratios, pub, labels, years, 1)
    print(f"행 {len(M):,} / 사건 {M.y.sum():,}", flush=True)

    orig, amend = dm.load_filings()
    aud = dm.load_auditors()
    f = fs.load_filings_all()
    cats = [c for c, _ in fs.CATS if not c.startswith("L_")]
    C = pd.concat([dm.features(M, 1, orig, amend, aud).reset_index(drop=True),
                   fs.event_features(M, 1, f, cats).reset_index(drop=True)], axis=1)
    C = C.loc[:, C.notna().any() & (C.nunique() > 1)]
    print(f"후보 {C.shape[1]}개", flush=True)

    # 선별 — 2013~2017 학습 구간, B + 후보, 학습 데이터 SHAP
    import shap
    ea.set_split("B")
    clf, Xtr, _, _ = fs.fit_predict(pd.concat([B, C], axis=1), M)
    sv = shap.TreeExplainer(clf).shap_values(Xtr)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=Xtr.columns)
    rank = imp[C.columns].sort_values(ascending=False)
    top = list(rank.index[:K])
    rank.rename("mean_abs_shap").to_csv("results/final4_select.csv", encoding="utf-8-sig")
    print("선별:", top, flush=True)

    scores, cis = [], []
    for split in ("A", "B"):
        ea.set_split(split)
        te = M.pred_year.isin(g.TEST_YEARS).values
        grp = M.loc[te, "corp_code"].reset_index(drop=True)
        conds = {"A": A, "A+": pd.concat([A, C[top]], axis=1),
                 "B": B, "B+": pd.concat([B, C[top]], axis=1)}
        P = {}
        for name, X in conds.items():
            _, _, p, y = fs.fit_predict(X, M)
            P[name] = p
            scores.append({"평가창": split, "조건": name, "사건": int(y.sum()),
                           **{k: round(float(v), 4) for k, v in dm.score(y, p).items()}})
            print(scores[-1], flush=True)
        for a, b, what in [("A", "B", "누출 효과"), ("B+", "B", "공시변수 기여"),
                           ("A", "B+", "남은 격차"), ("A+", "B+", "같은 변수 붙인 뒤 격차"),("A+", "A", "참고")]:
            for r in ea.boot_ci(y, P[a], P[b], grp, N_BOOT, rng):
                cis.append({"평가창": split, "비교": f"{a}−{b}", "의미": what, **r})

        # 공식 — 선별 10개 + KSCORE + O_SCORE, 학습 구간에서 계수 고정
        cols = ["KSCORE", "O_SCORE"] + top
        X = pd.concat([B, C[top]], axis=1)[cols]
        tr = M.pred_year.isin(g.TRAIN_YEARS).values
        lo, hi = X[tr].quantile(.01), X[tr].quantile(.99)
        lr = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                           LogisticRegression(max_iter=2000, class_weight="balanced"))
        lr.fit(X[tr].clip(lo, hi, axis=1), M.loc[tr, "y"])
        p = lr.predict_proba(X[te].clip(lo, hi, axis=1))[:, 1]
        s = dm.score(M.loc[te, "y"].values, p)
        scores.append({"평가창": split, "조건": "공식(로지스틱)", "사건": int(M.loc[te, "y"].sum()),
                       **{k: round(float(v), 4) for k, v in s.items()}})
        coef = pd.Series(lr[-1].coef_[0], index=cols).round(3)
        coef["(절편)"] = round(float(lr[-1].intercept_[0]), 3)
        coef.rename(f"창{split}_표준화계수").to_frame().T.to_csv(
            "results/final4_formula.csv", mode="a" if split == "B" else "w", encoding="utf-8-sig")
        print(scores[-1], "\n", coef.to_string(), flush=True)

    pd.DataFrame(scores).to_csv("results/final4_score.csv", index=False, encoding="utf-8-sig")
    ci = pd.DataFrame(cis)
    ci.to_csv("results/final4_ci.csv", index=False, encoding="utf-8-sig")
    print(ci.to_string())


if __name__ == "__main__":
    main()
