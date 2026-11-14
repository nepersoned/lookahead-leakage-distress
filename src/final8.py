"""본문 설계(final7)의 보강 — 모델 6종 · 공식 · 코스피.

    --models   코스닥 A · A+ · B · B+ 를 논문 6종 모델로 확장창 채점
    --formula  로지스틱 공식: 선별 변수 + KSCORE + O_SCORE (B+ 행동 변수만, 규정 경로 제외)
               계수는 선별 구간(2017~2018)에서 한 번 적합해 고정하고 2019~2024를 재적합 없이 채점
    --kospi    코스피 A−B (확장창 + 1·4월 시점). 코스피는 수시공시 목록이 없어 A·B만 잰다

선별 목록은 final7 --start 2017 결과 파일을 그대로 읽는다 (다시 뽑지 않는다).
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import final7 as f7  # 정규식 순서 수정이 import 시 적용된다
import disclosure_mock as dm
import expand_analysis as ea
import feature_screen as fs
import final5 as f5
import gate3e_full as g
import models6 as m6

warnings.filterwarnings("ignore")
START, TEST = 2017, list(range(2019, 2025))


def selected():
    top = list(pd.read_csv("results/final7_s2017_shap.csv", index_col=0).index[:10])
    sp = pd.read_csv("results/final7_s2017_sparse.csv")
    return top + [c for c in sp[sp.강제포함].변수 if c not in top]


def kosdaq_design():
    data = f7.load()
    ratios, pub, labels, mc, c2s = data
    A, B, M = f7.design(ratios, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    orig, amend = dm.load_filings()
    aud = dm.load_auditors()
    f = fs.load_filings_all()
    f["cat"] = fs.classify(f["report_nm"])
    D = dm.features(M, 1, orig, amend, aud).drop(columns=["amend_1y", "amend_3y"])
    Eb = fs.event_features(M, 1, f, f5.BEHAV).drop(columns=["amend_any_1y", "amend_any_3y", "amend_share_3y"])
    CB = pd.concat([D, Eb, f5.extra_features(M, f, orig)], axis=1).reset_index(drop=True)
    Er = fs.event_features(M, 1, f, f5.RULE).reset_index(drop=True)
    CR = Er[[c for c in Er.columns if c.startswith("L_")]]
    CR = CR.loc[:, CR.nunique() > 1]
    return A, B, M, CB, CR


def models(rng, boot):
    A, B, M, CB, CR = kosdaq_design()
    sel = selected()
    conds = {"A": A, "A+": pd.concat([A, CB[sel], CR], axis=1),
             "B": B, "B+": pd.concat([B, CB[sel], CR], axis=1)}
    te = M.pred_year.isin(TEST).values
    y = M.loc[te, "y"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)
    rows, cis = [], []
    for mname in m6.build_models():
        P = {}
        for cname, X in conds.items():
            p = np.full(len(M), np.nan)
            for yy in TEST:
                g.TRAIN_YEARS, g.TEST_YEARS = list(range(START, yy)), [yy]
                model = m6.build_models()[mname]
                tr = M.pred_year.isin(g.TRAIN_YEARS).values
                lo, hi = X[tr].quantile(.01), X[tr].quantile(.99)
                model.fit(X[tr].clip(lo, hi, axis=1), M.loc[tr, "y"].values)
                m = M.pred_year.values == yy
                p[m] = model.predict_proba(X[m].clip(lo, hi, axis=1))[:, 1]
            P[cname] = p[te]
            rows.append({"모델": mname, "조건": cname,
                         **{k: round(float(v), 4) for k, v in dm.score(y, P[cname]).items()}})
            print(rows[-1], flush=True)
        for a, b in [("A", "B"), ("B+", "B"), ("B+", "A"), ("A+", "B+")]:
            for r in ea.boot_ci(y, P[a], P[b], grp, boot, rng):
                if r["지표"] in ("PR-AUC", "AUC"):
                    cis.append({"모델": mname, "비교": f"{a}−{b}", **r})
    pd.DataFrame(rows).to_csv("results/final8_models.csv", index=False, encoding="utf-8-sig")
    ci = pd.DataFrame(cis)
    ci.to_csv("results/final8_models_ci.csv", index=False, encoding="utf-8-sig")
    print(ci.pivot_table(index="모델", columns=["비교", "지표"], values="격차").round(3).to_string())
    print(ci.pivot_table(index="모델", columns=["비교", "지표"], values="유의").to_string())


def formula():
    A, B, M, CB, CR = kosdaq_design()
    sel = selected()
    cols = ["KSCORE", "O_SCORE"] + sel
    X = pd.concat([B, CB[sel]], axis=1)[cols]
    tr = M.pred_year.isin([2017, 2018]).values
    te = M.pred_year.isin(TEST).values
    lo, hi = X[tr].quantile(.01), X[tr].quantile(.99)
    rows = []
    for tag, Cs in [("L2 C=1", 1.0), ("L2 C=0.1", 0.1)]:
        lr = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                           LogisticRegression(C=Cs, max_iter=5000, class_weight="balanced"))
        lr.fit(X[tr].clip(lo, hi, axis=1), M.loc[tr, "y"])
        p = lr.predict_proba(X[te].clip(lo, hi, axis=1))[:, 1]
        s = dm.score(M.loc[te, "y"].values, p)
        rows.append({"공식": tag, **{k: round(float(v), 4) for k, v in s.items()}})
        imp, sc = lr[0], lr[1]
        coef = pd.DataFrame({"변수": cols, "표준화계수": lr[-1].coef_[0].round(3),
                             "학습평균": sc.mean_.round(4), "학습표준편차": sc.scale_.round(4),
                             "원척도계수": (lr[-1].coef_[0] / sc.scale_).round(5)})
        coef.loc[len(coef)] = ["(절편)", round(float(lr[-1].intercept_[0]), 3), np.nan, np.nan,
                               round(float(lr[-1].intercept_[0] - (lr[-1].coef_[0] * sc.mean_ / sc.scale_).sum()), 4)]
        coef.to_csv(f"results/final8_formula_{tag.replace(' ', '').replace('=', '')}.csv",
                    index=False, encoding="utf-8-sig")
        print(tag, rows[-1], "\n", coef.to_string(), flush=True)
    # 비교 기준: 같은 변수·같은 고정 학습의 LightGBM, 그리고 KSCORE+O_SCORE만의 공식
    for tag, cs in [("K+O 점수만", ["KSCORE", "O_SCORE"])]:
        lr = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                           LogisticRegression(max_iter=5000, class_weight="balanced"))
        lr.fit(X[tr][cs].clip(lo[cs], hi[cs], axis=1), M.loc[tr, "y"])
        p = lr.predict_proba(X[te][cs].clip(lo[cs], hi[cs], axis=1))[:, 1]
        rows.append({"공식": tag, **{k: round(float(v), 4) for k, v in dm.score(M.loc[te, "y"].values, p).items()}})
    g.TRAIN_YEARS, g.TEST_YEARS = [2017, 2018], TEST
    _, _, p, yv = fs.fit_predict(X, M)
    rows.append({"공식": "LightGBM 고정학습(참고)", **{k: round(float(v), 4) for k, v in dm.score(yv, p).items()}})
    out = pd.DataFrame(rows)
    out.to_csv("results/final8_formula_score.csv", index=False, encoding="utf-8-sig")
    print(out.to_string())


def kospi(rng, boot):
    ratios, pub, labels = ea.load_market("KOSPI", "classic")
    o = ratios
    ratios["O_SCORE"] = (-1.32 - 0.407 * o.O_SIZE + 6.03 * o.O_TLTA - 1.43 * o.O_WCTA
                         + 0.0757 * o.O_CLCA - 1.72 * o.O_OENEG - 2.37 * o.O_NITA
                         - 1.83 * o.O_FUTL_CFO + 0.285 * o.O_INTWO - 0.521 * o.O_CHIN)
    _, _, _, mc, c2s = f7.load()
    rows, cis = [], []
    for month in (1, 2, 3, 4, 5):
        A, B, M = f7.design(ratios, pub, labels, mc, c2s, list(range(2017, 2025)), month)
        te = M.pred_year.isin(TEST).values
        y = M.loc[te, "y"].values
        pa, pb = f7.rolling(A, M, START, TEST), f7.rolling(B, M, START, TEST)
        rows.append({"월": month, "행": int(te.sum()), "사건": int(y.sum()),
                     "lag>0": round(float((M.loc[te, "lag"] > 0).mean()), 3),
                     **{f"A_{k}": round(float(v), 4) for k, v in dm.score(y, pa).items()},
                     **{f"B_{k}": round(float(v), 4) for k, v in dm.score(y, pb).items()}})
        for r in ea.boot_ci(y, pa, pb, M.loc[te, "corp_code"].reset_index(drop=True), boot, rng):
            cis.append({"월": month, **r})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/final8_kospi.csv", index=False, encoding="utf-8-sig")
    ci = pd.DataFrame(cis)
    ci.to_csv("results/final8_kospi_ci.csv", index=False, encoding="utf-8-sig")
    print(ci.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--formula", action="store_true")
    ap.add_argument("--kospi", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    if a.formula:
        formula()
    if a.models:
        models(rng, a.boot)
    if a.kospi:
        kospi(rng, a.boot)
