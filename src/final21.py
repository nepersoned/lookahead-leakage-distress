"""시장 미시구조 변수 — 가격·거래량·공매도로 만든다.

자료: data/market_own/release (src/market_collect.py로 KRX 날짜별 스냅샷 수집)
    OHLCV  2014-07-01~2024-12-30 일별, 코스피·코스닥 전 종목 스냅샷
    공매도 순보유잔고 2016-06-30~

예측 시점 T = 그해 1월 1일. **T 직전 거래일까지의 자료만** 쓴다.

변수
    ret_3m / ret_6m / ret_12m   누적 수익률
    vol_3m                      일간 수익률 표준편차
    amihud_3m                   비유동성 (|수익률| / 거래대금)
    turn_3m                     회전율 (거래량 / 상장주식수)
    vol_spike                   최근 1개월 거래량 / 직전 6개월 평균
    penny                       종가 1,000원 미만 여부
    mcap_log                    시가총액 로그
    max_drawdown_12m            1년 최대 낙폭
    short_ratio                 공매도 잔고 / 상장주식수 (금지 기간은 결측)
    short_chg_3m                공매도 잔고 비율 변화

출력: results/final21_score.csv, final21_ci.csv
"""
from __future__ import annotations

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import final7 as f7
import final8 as f8
import disclosure_mock as dm
import expand_analysis as ea
from dart import Dart

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
REL = ROOT / "data" / "market_own" / "release"
RES = ROOT / "results"
TEST, START = f8.TEST, f8.START


def load_market(kind: str, years) -> pd.DataFrame:
    frames = []
    for y in years:
        for mkt in ("KOSDAQ", "KOSPI"):
            p = REL / f"market_{kind}_{mkt}_{y}.csv.gz"
            if p.exists():
                frames.append(pd.read_csv(p, dtype={"ticker": str, "date": str}))
    d = pd.concat(frames, ignore_index=True)
    d["ticker"] = d.ticker.str.zfill(6)
    d["dt"] = pd.to_datetime(d.date, format="%Y%m%d")
    return d


def features(M, ohlcv, short, c2s):
    px = ohlcv.rename(columns={c: c.lower() for c in ohlcv.columns})
    cols = {c: c for c in px.columns}
    close = next(c for c in px.columns if c in ("종가", "close"))
    vol = next(c for c in px.columns if c in ("거래량", "volume"))
    val = next((c for c in px.columns if c in ("거래대금", "value")), None)
    cap = next((c for c in px.columns if c in ("시가총액", "mcap")), None)
    shares = next((c for c in px.columns if c in ("상장주식수", "shares")), None)
    px = px[["ticker", "dt", close, vol] + [c for c in (val, cap, shares) if c]]
    px = px.rename(columns={close: "close", vol: "vol", val: "val",
                            cap: "mcap", shares: "shares"})
    px = px[px.close > 0].sort_values("dt")
    by_tic = {t: g for t, g in px.groupby("ticker")}

    sh = None
    if short is not None and len(short):
        s = short.rename(columns={c: c.lower() for c in short.columns})
        bal = next((c for c in s.columns if "잔고수량" in c or "balance" in c), None)
        if bal:
            s = s[["ticker", "dt", bal]].rename(columns={bal: "sbal"})
            sh = {t: g.sort_values("dt") for t, g in s.groupby("ticker")}

    rows = []
    for code, y in zip(M.corp_code, M.pred_year):
        tic = c2s.get(code)
        T = pd.Timestamp(year=y, month=1, day=1)
        r = {}
        g = by_tic.get(tic) if tic else None
        if g is not None:
            w = g[g.dt < T]
            if len(w) > 20:
                c = w.close.to_numpy(float)
                ret = np.diff(np.log(c))
                for lab, n in (("3m", 62), ("6m", 124), ("12m", 248)):
                    if len(c) > n:
                        r[f"ret_{lab}"] = float(c[-1] / c[-n] - 1)
                if len(ret) > 62:
                    r["vol_3m"] = float(np.std(ret[-62:]) * np.sqrt(252))
                    if "val" in w:
                        v = w.val.to_numpy(float)[-62:]
                        with np.errstate(divide="ignore", invalid="ignore"):
                            am = np.abs(ret[-61:]) / np.where(v[1:] > 0, v[1:], np.nan)
                        r["amihud_3m"] = float(np.nanmean(am) * 1e9)
                if "shares" in w and len(w) > 62:
                    sh_n = w.shares.to_numpy(float)[-62:]
                    vv = w.vol.to_numpy(float)[-62:]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r["turn_3m"] = float(np.nanmean(np.where(sh_n > 0, vv / sh_n, np.nan)))
                if len(w) > 145:
                    v1 = w.vol.to_numpy(float)[-21:].mean()
                    v6 = w.vol.to_numpy(float)[-145:-21].mean()
                    r["vol_spike"] = float(v1 / v6) if v6 > 0 else np.nan
                r["penny"] = float(c[-1] < 1000)
                if "mcap" in w:
                    m = float(w.mcap.iloc[-1])
                    r["mcap_log"] = float(np.log(m)) if m > 0 else np.nan
                if len(c) > 248:
                    cc = c[-248:]
                    r["max_dd_12m"] = float(cc.min() / np.maximum.accumulate(cc).max() - 1)
        if sh is not None and tic in (sh or {}):
            sw = sh[tic]
            sw = sw[sw.dt < T]
            if len(sw) and g is not None and "shares" in g:
                gs = g[g.dt < T]
                if len(gs):
                    shr = float(gs.shares.iloc[-1])
                    if shr > 0:
                        r["short_ratio"] = float(sw.sbal.iloc[-1] / shr)
                        if len(sw) > 62:
                            r["short_chg_3m"] = float((sw.sbal.iloc[-1] - sw.sbal.iloc[-62]) / shr)
        rows.append(r)
    return pd.DataFrame(rows, index=M.index)


def main():
    rng = np.random.default_rng(42)
    A, B, M, CB, CR = f8.kosdaq_design()
    sel = f8.selected()
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes.stock_code.astype(str).str.strip().str.zfill(6)
    c2s = (codes[codes.stock_code.str.len() == 6].drop_duplicates("corp_code")
           .set_index("corp_code")["stock_code"].to_dict())

    ohlcv = load_market("ohlcv", range(2014, 2025))
    try:
        short = load_market("short", range(2016, 2025))
    except Exception:
        short = None
    print(f"OHLCV {len(ohlcv):,}행 / 공매도 {0 if short is None else len(short):,}행", flush=True)
    print("열:", list(ohlcv.columns), flush=True)

    F = features(M, ohlcv, short, c2s).reset_index(drop=True)
    te = M.pred_year.isin(TEST).values
    print(f"\n변수 {F.shape[1]}개 / 채점 결측\n{(F[te].isna().mean() * 100).round(1).to_string()}",
          flush=True)
    tr = M.pred_year.isin([2017, 2018]).values
    cmp_ = pd.DataFrame({"사건": F[tr & (M.y == 1).values].mean(),
                         "정상": F[tr & (M.y == 0).values].mean()}).round(3)
    cmp_["배수"] = (cmp_.사건 / cmp_.정상.replace(0, np.nan)).round(2)
    print("\n=== 학습 구간 비교 ===\n" + cmp_.to_string(), flush=True)

    base = pd.concat([B, CB[sel], CR], axis=1)
    conds = {"B+ (기준)": base, "A+": pd.concat([A, CB[sel], CR], axis=1),
             "+ 시장 변수": pd.concat([base, F], axis=1),
             "A+ + 시장 변수": pd.concat([A, CB[sel], CR, F], axis=1),
             "시장 변수만 (B + 시장)": pd.concat([B, F], axis=1)}
    y = M.loc[te, "y"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)
    P, rows = {}, []
    for name, X in conds.items():
        P[name] = f7.rolling(X, M, START, TEST)
        rows.append({"조건": name, "변수": X.shape[1],
                     **{k: round(float(v), 4) for k, v in dm.score(y, P[name]).items()}})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "final21_score.csv", index=False, encoding="utf-8-sig")
    cis = []
    for a, b, what in [("+ 시장 변수", "B+ (기준)", "시장 변수 기여"),
                       ("A+ + 시장 변수", "+ 시장 변수", "남는 격차"),
                       ("+ 시장 변수", "A+", "A+를 넘는가")]:
        for r in ea.boot_ci(y, P[a], P[b], grp, 1000, rng):
            cis.append({"비교": f"{a} − {b}", "의미": what, **r})
    ci = pd.DataFrame(cis)
    ci.to_csv(RES / "final21_ci.csv", index=False, encoding="utf-8-sig")
    print(ci.to_string())


if __name__ == "__main__":
    main()
