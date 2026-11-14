"""final44 — 고전 해저드 모형 기준선에서도 누출이 나타나는가 (탐색적).

Shumway(2001)의 이산시간 해저드 모형(기업·연도 합동 로짓)을 기준선으로 둔다.
    회계 변수  NI/TA(O_NITA), TL/TA(O_TLTA)          ← A(결산일 정렬)와 B(접수일 정렬)에서 다르게 붙는다
    시장 변수  상대 규모 log(시총/시장 전체 시총), 초과수익률(직전 12개월, 코스닥 동일가중 대비),
               수익률 변동성 σ(직전 3개월 일별 로그수익률 표준편차, 연율)  ← T 이전 자료만, A·B 공통
모형: 가중치 없는 L2 로지스틱(중앙값 대치·표준화). 1·99% 분위수 절단은 학습 구간 기준. 확장 창, 1월 예측.
비교: Shumway-A vs Shumway-B, 그리고 회계 변수만(Altman·Ohlson식 로짓) A vs B. 본문 LightGBM 결과와 나란히 본다.
출력: results/final44_score.csv, final44_ci.csv, final44.log
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import final7 as f7
import final21 as f21
import disclosure_mock as dm
import expand_analysis as ea
from dart import Dart

warnings.filterwarnings("ignore")
START, TEST = 2017, list(range(2019, 2025))


def market_vars(M, c2s):
    px = f21.load_market("ohlcv", range(2014, 2025))
    px = px.rename(columns={c: c.lower() for c in px.columns})
    close = next(c for c in px.columns if c in ("종가", "close"))
    cap = next((c for c in px.columns if c in ("시가총액", "mcap")), None)
    px = px[["ticker", "dt", close, cap]].rename(columns={close: "close", cap: "mcap"})
    px = px[px.close > 0].sort_values(["ticker", "dt"])
    px["r"] = px.groupby("ticker").close.transform(lambda s: np.log(s).diff())
    mkt_r = px.groupby("dt").r.mean()          # 동일가중 시장 수익률
    tot_cap = px.groupby("dt").mcap.sum()
    by = {t: g for t, g in px.groupby("ticker")}
    rows = []
    for code, y in zip(M.corp_code, M.pred_year):
        T = pd.Timestamp(year=y, month=1, day=1)
        g = by.get(c2s.get(code))
        r = {"RSIZE": np.nan, "EXRET": np.nan, "SIGMA": np.nan}
        if g is not None:
            w = g[g.dt < T]
            if len(w) > 62:
                last = w.iloc[-1]
                if last["mcap"] > 0:
                    r["RSIZE"] = float(np.log(last["mcap"] / tot_cap.loc[last["dt"]]))
                rr = w.r.to_numpy(float)
                r["SIGMA"] = float(np.nanstd(rr[-62:]) * np.sqrt(252))
                w12 = w[w.dt >= T - pd.DateOffset(years=1)]
                if len(w12) > 120:
                    r["EXRET"] = float(np.nansum(w12.r) - mkt_r.loc[w12.dt].sum())
        rows.append(r)
    return pd.DataFrame(rows)


def logit_rolling(X, M):
    yr, yv = M.pred_year.values, M.y.values
    p = np.full(len(M), np.nan)
    for Y in TEST:
        tr, te = (yr >= START) & (yr < Y), yr == Y
        lo, hi = X[tr].quantile(.01), X[tr].quantile(.99)
        Xc = X.clip(lo, hi, axis=1)
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(Xc[tr], yv[tr])
        p[te] = m.predict_proba(Xc[te])[:, 1]
    return p[np.isin(yr, TEST)]


def main():
    rng = np.random.default_rng(42)
    _, pub, labels, mc, c2s = f7.load()
    r = pd.read_csv(ea.FULL / "classic_orig.csv", dtype={"corp_code": str})
    r["corp_code"] = r.corp_code.str.zfill(8)
    A, B, M = f7.design(r, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    A, B, M = A.reset_index(drop=True), B.reset_index(drop=True), M.reset_index(drop=True)
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes.stock_code.astype(str).str.strip().str.zfill(6)
    c2s2 = (codes[codes.stock_code.str.len() == 6].drop_duplicates("corp_code")
            .set_index("corp_code")["stock_code"].to_dict())
    MV = market_vars(M, c2s2)
    print("시장 변수 결측:", MV.isna().mean().round(3).to_dict(), flush=True)
    acc = ["O_NITA", "O_TLTA"]
    X = {"Shumway-A": pd.concat([A[acc], MV], axis=1), "Shumway-B": pd.concat([B[acc], MV], axis=1),
         "회계로짓-A": A[["O_NITA", "O_TLTA", "O_WCTA", "K_RETA", "O_CLCA", "O_SIZE"]],
         "회계로짓-B": B[["O_NITA", "O_TLTA", "O_WCTA", "K_RETA", "O_CLCA", "O_SIZE"]],
         "시장만": MV}
    P = {k: logit_rolling(v, M) for k, v in X.items()}
    te = M.pred_year.isin(TEST).values
    y = M.loc[te, "y"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)
    score = pd.DataFrame([{"조건": k, **{a: round(float(b), 4) for a, b in dm.score(y, p).items()}} for k, p in P.items()])
    cis = []
    for a, b in [("Shumway-A", "Shumway-B"), ("회계로짓-A", "회계로짓-B")]:
        for rr in ea.boot_ci(y, P[a], P[b], grp, 1000, rng):
            f_ = rr["0초과"]
            cis.append({"비교": f"{a} − {b}", **rr, "p(양측)": round(max(2 * min(f_, 1 - f_), 0.002), 3)})
    cis = pd.DataFrame(cis)
    score.to_csv("results/final44_score.csv", index=False, encoding="utf-8-sig")
    cis.to_csv("results/final44_ci.csv", index=False, encoding="utf-8-sig")
    txt = score.to_string(index=False) + "\n\n" + cis.to_string(index=False)
    open("results/final44.log", "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
