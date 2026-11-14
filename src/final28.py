"""사전에 정한 두 가지 보강을 한 번만 잰다 — 결과와 무관하게 그대로 보고한다.

계획 확정: 2026-10-03, 채점 결과를 보기 전. 이 파일의 설계는 실행 뒤 바꾸지 않는다.

배경
    final27(원문판 + 고친 모집단)에서 B+시장 − A = +0.049, 양측 p = 0.058로 유의 경계였다.
    아래 두 보강은 그 결과를 보고 떠올린 것이지만, 둘 다 채점 구간을 쓰지 않고 정하며
    한 번만 실행한다. 몇 번 시도했는지(1회)와 함께 보고한다.

보강 1 — 지정 이력 변수 (T 이전 공개 정보)
    hist_any   T 이전 관리종목 지정 경험 여부
    hist_n     T 이전 지정 횟수 (기업·연도 단위)
    hist_yrs   마지막 지정 이후 경과 연수 (없으면 결측)
    공시 변수 묶음에 넣는다 → B+ · A+에 같이 들어가고 재무만 쓰는 A·B에는 없다.
    모집단 수정(final27) 전에는 이 정보가 라벨과 사실상 같았으나, 수정 뒤에는 재지정률 8.3%의
    정상적인 위험 신호다.

보강 2 — 하이퍼파라미터를 학습 구간 안에서 고른다
    채점연도 Y마다 2017~(Y−2)로 학습, (Y−1)로 검증해 검증 PR-AUC가 가장 높은 설정을 고르고,
    그 설정으로 2017~(Y−1) 전체를 학습해 Y를 채점한다. 모든 조건에 같은 절차.
    격자: num_leaves {7, 15, 31} × min_child_samples {10, 30} × reg_lambda {0, 5} (12개)
    나머지(300 trees, lr 0.05, balanced, 1·99% 절단)는 본문과 같다.

조건 (원문판 재무 · final27의 선별 목록 · 시장 10개)
    A · B · B+ · B+시장 · A+ · A+시장   ×   {고정 설정, 학습 구간 내 선택}
    B+·A+는 이력 변수 포함

출력: results/final28_score.csv, final28_ci.csv, final28_params.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import final7 as f7
import final21 as f21
import final27 as f27
import disclosure_mock as dm
import expand_analysis as ea
from dart import Dart

warnings.filterwarnings("ignore")
START, TEST = 2017, list(range(2019, 2025))
GRID = [dict(num_leaves=nl, min_child_samples=mc, reg_lambda=rl)
        for nl in (7, 15, 31) for mc in (10, 30) for rl in (0, 5)]
BASE = dict(num_leaves=15, min_child_samples=10, reg_lambda=0)


def fit(X, y, tr, te, params):
    import lightgbm as lgb
    lo, hi = X[tr].quantile(.01), X[tr].quantile(.99)
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, class_weight="balanced",
                             verbose=-1, random_state=42, **params)
    clf.fit(X[tr].clip(lo, hi, axis=1), y[tr])
    return clf.predict_proba(X[te].clip(lo, hi, axis=1))[:, 1]


def rolling(X, M, tune, log, name):
    yv, yr = M.y.values, M.pred_year.values
    p = np.full(len(M), np.nan)
    for Y in TEST:
        params = BASE
        if tune:
            tr_i, va = (yr >= START) & (yr <= Y - 2), yr == Y - 1
            best = max(GRID, key=lambda pr: dm.score(yv[va], fit(X, yv, tr_i, va, pr))["PR-AUC"])
            params = best
            log.append({"조건": name, "채점연도": Y, **best})
        tr, te = (yr >= START) & (yr < Y), yr == Y
        p[te] = fit(X, yv, tr, te, params)
    return p[np.isin(yr, TEST)]


def history(M, labels):
    lab = labels.groupby("corp_code")["event_dt"].apply(list).to_dict()
    rows = []
    for c, y in zip(M.corp_code, M.pred_year):
        T = pd.Timestamp(year=y, month=1, day=1)
        past = [d for d in lab.get(c, []) if d < T]
        rows.append({"hist_any": float(bool(past)), "hist_n": float(len(past)),
                     "hist_yrs": (T - max(past)).days / 365.25 if past else np.nan})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(42)
    _, pub, labels, mc, c2s = f7.load()
    ratios = pd.read_csv("data/full/classic_orig.csv", dtype={"corp_code": str})
    ratios["corp_code"] = ratios.corp_code.str.zfill(8)
    A, B, M = f7.design(ratios, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    te = M.pred_year.isin(TEST).values
    y = M.loc[te, "y"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)

    CB = f27.disclosure(M)
    top = list(pd.read_csv("results/final27_shap.csv")["변수"][:10])
    sp = pd.read_csv("results/final27_sparse.csv")
    sel = top + [c for c in sp[sp.강제포함].변수 if c not in top]
    H = history(M, labels)
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes.stock_code.astype(str).str.strip().str.zfill(6)
    c2s2 = (codes[codes.stock_code.str.len() == 6].drop_duplicates("corp_code")
            .set_index("corp_code")["stock_code"].to_dict())
    F = f21.features(M, f21.load_market("ohlcv", range(2014, 2025)), None, c2s2).reset_index(drop=True)

    bp = pd.concat([B, CB[sel], H], axis=1)
    ap = pd.concat([A, CB[sel], H], axis=1)
    conds = {"A": A, "B": B, "B+": bp, "B+시장": pd.concat([bp, F], axis=1),
             "A+": ap, "A+시장": pd.concat([ap, F], axis=1)}
    rows, cis, log = [], [], []
    for tune in (False, True):
        tag = "학습 구간 내 선택" if tune else "고정 설정"
        P = {}
        for k, X in conds.items():
            P[k] = rolling(X, M, tune, log, k)
            rows.append({"설정": tag, "조건": k, **{a: round(float(b), 4) for a, b in dm.score(y, P[k]).items()}})
            print(rows[-1], flush=True)
        for a, b, what in [("A", "B", "누출"), ("B+", "B", "공시+이력 복구"), ("B+", "A", "B+가 A를 넘는가"),
                           ("B+시장", "A", "B+시장이 A를 넘는가"), ("A+시장", "B+시장", "남는 격차 (시장까지)")]:
            for r in ea.boot_ci(y, P[a], P[b], grp, 1000, rng):
                f_ = r["0초과"]
                cis.append({"설정": tag, "비교": f"{a} − {b}", "의미": what, **r,
                            "p(양측)": round(max(2 * min(f_, 1 - f_), 0.002), 3)})
    pd.DataFrame(rows).to_csv("results/final28_score.csv", index=False, encoding="utf-8-sig")
    ci = pd.DataFrame(cis)
    ci.to_csv("results/final28_ci.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(log).to_csv("results/final28_params.csv", index=False, encoding="utf-8-sig")
    print(ci[ci.지표 == "PR-AUC"].to_string(), flush=True)


if __name__ == "__main__":
    main()
