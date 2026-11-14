"""경제적 의미 — 모형이 고른 위험 상위 기업을 피하면 무엇이 달라지나.

입력: results/final27_preds.csv (채점 행별 A·B·B+·A+·B+시장·A+시장 예측값, T = 1월 1일)
      data/market_own/release/market_ohlcv_KOSDAQ_*.csv.gz (KRX Open API 일별매매정보)

수익률: T 직전 거래일 종가에서 1년 뒤 직전 거래일 종가까지의 보유수익률.
    일별 수익률을 이어 붙이며, 상장주식수가 1.5배 이상 늘거나 2/3 이하로 줄고 가격이 그 반대로
    움직인 날(분할·병합·감자)은 주식수 비율로 보정한다. 유상증자 등 나머지 주식수 변화는 보정하지 않는다.
    기간 중 상장폐지되면 마지막 거래일(정리매매 포함)까지의 수익률을 쓴다.

보는 것 (연도별 → 2019-2024 평균)
    ① 상위 k개(50·100) 위험 기업의 평균 보유수익률과 그중 관리종목 지정 비율
    ② 전체 동일가중 수익률 대비, 상위 k개를 뺀 포트폴리오의 초과수익률 (피한 손실)
    ③ 모형 간 차이 (A − B: 누출이 만든 '가짜' 이득, B+시장 − B 등)

출력: results/final33_topk.csv, final33_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import final21 as f21
from dart import Dart

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
KS = (50, 100)


def holding_returns(px: pd.DataFrame, years) -> pd.DataFrame:
    """(ticker, year) → T=1/1 직전 종가부터 1년 보유수익률. 분할·병합 보정."""
    px = px.sort_values(["ticker", "dt"])
    px = px[px["종가"] > 0]
    g = px.groupby("ticker")
    p0, s0 = g["종가"].shift(), g["상장주식수"].shift()
    r = px["종가"] / p0
    s = px["상장주식수"] / s0
    split = ((s >= 1.5) | (s <= 2 / 3)) & ((r * s).between(0.5, 2.0))
    r = np.where(split, r * s, r)
    px = px.assign(lr=np.log(pd.Series(r, index=px.index).where(p0.notna(), 1.0)))
    rows = []
    for y in years:
        T0, T1 = pd.Timestamp(year=y, month=1, day=1), pd.Timestamp(year=y + 1, month=1, day=1)
        w = px[(px.dt < T1)]
        start = w[w.dt < T0].groupby("ticker").dt.max()          # T 직전 거래일
        win = w[w.dt >= T0]
        win = win[win.ticker.isin(start.index)]
        agg = win.groupby("ticker").agg(lr=("lr", "sum"), last=("dt", "max"))
        agg["ret"] = np.exp(agg.lr) - 1
        agg["delisted_or_halted"] = agg["last"] < (T1 - pd.Timedelta(days=10))
        rows.append(agg.reset_index().assign(year=y)[["ticker", "year", "ret", "delisted_or_halted"]])
    return pd.concat(rows, ignore_index=True)


def main():
    P = pd.read_csv(OUT / "final27_preds.csv", dtype={"corp_code": str})
    P["corp_code"] = P.corp_code.str.zfill(8)
    models = [c for c in P.columns if c not in ("corp_code", "pred_year", "y")]
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes.stock_code.astype(str).str.strip().str.zfill(6)
    c2s = (codes[codes.stock_code.str.len() == 6].drop_duplicates("corp_code")
           .set_index("corp_code")["stock_code"])
    P["ticker"] = P.corp_code.map(c2s)

    px = f21.load_market("ohlcv", range(2018, 2026))
    R = holding_returns(px, sorted(P.pred_year.unique()))
    D = P.merge(R, left_on=["ticker", "pred_year"], right_on=["ticker", "year"], how="left")
    print(f"수익률 매칭 {D.ret.notna().mean():.1%} / 행 {len(D):,}", flush=True)
    D = D[D.ret.notna()]
    D["ret"] = D.ret.clip(-1, 5)            # 극단 상승(5배 초과)만 자른다

    rows = []
    for y, g in D.groupby("pred_year"):
        allr = g.ret.mean()
        for m in models:
            o = g.sort_values(m, ascending=False)
            for k in KS:
                top, rest = o.head(k), o.iloc[k:]
                rows.append({"연도": y, "모형": m, "k": k, "상위k 평균수익률": top.ret.mean(),
                             "상위k 지정비율": top.y.mean(), "상위k 거래중단·폐지비율": top.delisted_or_halted.mean(),
                             "전체 평균수익률": allr, "제외 포트폴리오 초과수익률": rest.ret.mean() - allr})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "final33_topk.csv", index=False, encoding="utf-8-sig")
    S = (T.groupby(["모형", "k"])[["상위k 평균수익률", "상위k 지정비율", "상위k 거래중단·폐지비율",
                                  "전체 평균수익률", "제외 포트폴리오 초과수익률"]].mean().round(4).reset_index())
    S.to_csv(OUT / "final33_summary.csv", index=False, encoding="utf-8-sig")
    print(S.to_string(index=False), flush=True)
    # 연도별 차이의 부호 일관성 (연도 6개라 검정 대신 방향만 본다)
    for a, b in [("A", "B"), ("B+시장", "B"), ("B+시장", "A")]:
        if a in models and b in models:
            for k in KS:
                d = (T[(T.모형 == a) & (T.k == k)].set_index("연도")["상위k 평균수익률"]
                     - T[(T.모형 == b) & (T.k == k)].set_index("연도")["상위k 평균수익률"])
                print(f"{a} − {b} (k={k}) 상위k 수익률 차이: 평균 {d.mean():+.3f}, 음수인 해 {int((d < 0).sum())}/{len(d)}")


if __name__ == "__main__":
    main()
