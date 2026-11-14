"""시가총액 월말 스냅샷 — KRX 정보데이터시스템 (pykrx).

날짜별 전 종목 스냅샷을 받으므로 그날 상장돼 있던 기업은 나중에 폐지됐어도
들어 있다. 네이버 등 종목별 시계열 소스는 폐지 종목이 없어 생존편향이 생긴다
(확인 결과 폐지 종목 3개 전부 0건).

쓰임 — K-Score의 시가총액/총부채 항, Altman Z 원판의 X4, 시장 규모 통제.
주의 — 관리종목 지정 사유에 시가총액 미달이 있으므로 그 사유 건은 따로 본다.

로그인이나 인증키가 필요 없다. pykrx가 공개 화면을 읽는다.
출력: data/full/marketcap.csv (date, market, ticker, close, mcap, volume, value, shares)
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"


def main():
    from pykrx import stock

    out = FULL / "marketcap.csv"
    done = set()
    parts = []
    if out.exists():
        prev = pd.read_csv(out, dtype={"ticker": str})
        parts.append(prev)
        done = set(zip(prev.date.astype(str), prev.market))

    for me in pd.date_range("2014-12-01", "2024-12-01", freq="MS") + pd.offsets.MonthEnd(0):
        d = stock.get_nearest_business_day_in_a_week(me.strftime("%Y%m%d"), prev=True)
        for mkt in ("KOSDAQ", "KOSPI"):
            if (d, mkt) in done:
                continue
            for attempt in range(4):
                try:
                    df = stock.get_market_cap(d, market=mkt)
                    break
                except Exception:
                    time.sleep(5 * (attempt + 1))
            else:
                print(f"  실패 {d} {mkt}", flush=True)
                continue
            df = df.reset_index().rename(columns={
                "티커": "ticker", "종가": "close", "시가총액": "mcap", "거래량": "volume",
                "거래대금": "value", "상장주식수": "shares"})
            df["date"], df["market"] = d, mkt
            parts.append(df)
            print(f"  {d} {mkt} {len(df):,}개사", flush=True)
            time.sleep(1)
        pd.concat(parts, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")

    allp = pd.concat(parts, ignore_index=True)
    print(f"\n시가총액 {len(allp):,}건 · {allp.date.nunique()}개 날짜 → marketcap.csv")


if __name__ == "__main__":
    main()
