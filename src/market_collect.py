"""KRX Open API로 날짜별 전 종목 일별매매정보를 받는다 — 시장 변수의 원자료.

공식 경로(openapi.krx.co.kr, 인증키는 `.env`의 KRX_API_KEY)만 쓴다. 서비스 두 개의 이용 승인이 필요하다.

    ksq_bydd_trd   코스닥 일별매매정보
    stk_bydd_trd   유가증권 일별매매정보

날짜별 스냅샷이라 이미 상장폐지된 종목도 그날 상장돼 있었다면 들어온다(생존편향 없음).
주말은 부르지 않고, 휴장일은 빈 표가 오므로 표시만 남긴다. 호출 사이에 쉬어 하루 한도(10,000건)
안에서 천천히 받는다. 받은 날짜는 건너뛰므로 다시 실행해도 된다.

사용
    python src/market_collect.py            # 2014-07-01 ~ 2024-12-31 수집
    python src/market_collect.py --pack     # 연도별 배포 파일로 묶기
출력
    data/market_own/daily/cap_{market}_{YYYYMMDD}.csv
    data/market_own/release/market_ohlcv_{market}_{YYYY}.csv.gz
      열: date · market · ticker · 종가 · 시가총액 · 거래량 · 거래대금 · 상장주식수
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "market_own"
DAILY, REL = OUT / "daily", OUT / "release"
START, END = pd.Timestamp("2014-07-01"), pd.Timestamp("2024-12-31")
API = "https://data-dbg.krx.co.kr/svc/apis/sto/"
SERVICES = {"KOSDAQ": "ksq_bydd_trd"}      # 분석 모집단이 코스닥이라 코스피(stk_bydd_trd)는 받지 않는다
COLS = {"ISU_CD": "ticker", "TDD_CLSPRC": "종가", "MKTCAP": "시가총액", "ACC_TRDVOL": "거래량",
        "ACC_TRDVAL": "거래대금", "LIST_SHRS": "상장주식수"}
PAUSE = 1.0          # 호출 간격(초)


def call(key: str, svc: str, ds: str, tries: int = 4) -> list:
    for i in range(tries):
        try:
            r = requests.get(API + svc, params={"basDd": ds}, headers={"AUTH_KEY": key}, timeout=60)
            if r.status_code == 200:
                return r.json().get("OutBlock_1") or []
            print(f"  {svc} {ds} 응답 {r.status_code}: {r.text[:120]}", flush=True)
            if r.status_code in (401, 403):
                raise SystemExit("인증 실패 — 서비스 이용 승인과 KRX_API_KEY를 확인한다")
        except requests.RequestException as e:
            print(f"  {svc} {ds} 재시도 {i + 1}: {type(e).__name__}", flush=True)
        time.sleep(10 * (i + 1))
    raise RuntimeError(f"{svc} {ds} 실패")


def collect():
    load_dotenv(ROOT / ".env")
    key = os.environ["KRX_API_KEY"]
    DAILY.mkdir(parents=True, exist_ok=True)
    days = [d for d in pd.date_range(START, END) if d.weekday() < 5]
    n_new = 0
    for d in days:
        ds = d.strftime("%Y%m%d")
        for m, svc in SERVICES.items():
            f, hol = DAILY / f"cap_{m}_{ds}.csv", DAILY / f"holiday_{m}_{ds}"
            if f.exists() or hol.exists():
                continue
            rows = call(key, svc, ds)
            time.sleep(PAUSE)
            if not rows:
                hol.touch()
                continue
            df = pd.DataFrame(rows)[list(COLS)].rename(columns=COLS)
            for c in ("종가", "시가총액", "거래량", "거래대금", "상장주식수"):
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
            df.to_csv(f, index=False, encoding="utf-8-sig")
            n_new += 1
        if d.day <= 3 and d.weekday() == 0 or d == days[-1]:
            print(f"{ds} 진행 중 (새로 받은 시장·일 {n_new})", flush=True)
    print(f"끝 — 새로 받은 시장·일 {n_new}", flush=True)


def pack():
    REL.mkdir(parents=True, exist_ok=True)
    for m in SERVICES:
        by_year: dict[str, list] = {}
        for f in sorted(DAILY.glob(f"cap_{m}_*.csv")):
            by_year.setdefault(f.stem[-8:-4], []).append(f)
        for y, fs in sorted(by_year.items()):
            parts = []
            for f in fs:
                df = pd.read_csv(f, dtype={"ticker": str})
                df.insert(0, "date", f.stem[-8:])
                df.insert(1, "market", m)
                parts.append(df)
            d = pd.concat(parts, ignore_index=True)
            d["ticker"] = d["ticker"].str.zfill(6)
            out = REL / f"market_ohlcv_{m}_{y}.csv.gz"
            d.to_csv(out, index=False, encoding="utf-8-sig", compression="gzip")
            print(f"{out.name}: {len(d):,}행 · {d['date'].nunique()}일", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", action="store_true")
    a = ap.parse_args()
    pack() if a.pack else collect()
