"""변수 세트 3 — 부실예측의 세계 표준 모형 두 개.

변수 세트 2(윤양현 외 2022의 21개)는 논문 한 편의 명세라 "왜 그 21개냐"는 질문에
답이 약하다. 그래서 누구도 반박하지 않는 기준선으로 한 번 더 잰다.

    Altman Z''  (Altman 1983/1995)  비제조·비상장용. 시가총액 대신 장부가 자본을 쓴다
        X1 운전자본/총자산  X2 이익잉여금/총자산  X3 EBIT/총자산  X4 장부가자본/총부채
        Z'' = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4

    Ohlson O    (Ohlson 1980)
        SIZE log(총자산)*   TLTA 총부채/총자산   WCTA 운전자본/총자산   CLCA 유동부채/유동자산
        OENEG 총부채>총자산  NITA 순이익/총자산    FUTL 영업현금흐름/총부채**
        INTWO 2년 연속 순손실  CHIN (NIt−NIt−1)/(|NIt|+|NIt−1|)

    *  원식은 GNP 물가지수로 나눈다. 연도 고정효과와 겹치므로 생략한다
    ** 원식은 영업자금(FFO)이다. DART에 없어 영업활동현금흐름으로 대신한다
    EBIT는 영업이익으로 근사한다

주가를 쓰는 원판 Altman Z는 일부러 뺐다. 이 프로젝트는 공시 자료만으로 잰다.

API 재호출은 없다 — vars21 수집 때 FY2014~2023 재무제표가 캐시에 들어 있다.
출력: data/full/classic_full.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dart import Dart, DartError
import vars21 as v21

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"

EXTRA = {
    "cur_assets": ("ifrs-full_CurrentAssets", ["유동자산"]),
    "cur_liab": ("ifrs-full_CurrentLiabilities", ["유동부채"]),
    "retained": ("ifrs-full_RetainedEarningsAccumulatedLosses",
                 ["이익잉여금", "이익잉여금(결손금)", "결손금"]),
    "depr": ("dart_AdjustmentsForDepreciationExpense", ["감가상각비"]),
}
# 기준선(고정) — K-Score와 Ohlson O. 이인로(2015) 인용 계수
BASE_COLS = ["K_lnTA", "K_lnSLTA", "K_RETA", "K_METL", "KSCORE",
             "O_SIZE", "O_TLTA", "O_WCTA", "O_CLCA", "O_OENEG", "O_NITA", "O_FUTL",
             "O_INTWO", "O_CHIN", "O_SCORE"]
COLS = ["Z_X1", "Z_X2", "Z_X3", "Z_X4", "Z2", "Z_X5", "ZSCORE",
        "K_lnTA", "K_lnSLTA", "K_RETA", "K_METL", "KSCORE", "O_FUTL_CFO",
        "O_SIZE", "O_TLTA", "O_WCTA", "O_CLCA", "O_OENEG", "O_NITA", "O_FUTL",
        "O_INTWO", "O_CHIN", "O_SCORE"]


def extract(items):
    base = v21.extract(items)
    by_id, by_nm = {}, {}
    for it in items:
        v = v21._num(it.get("thstrm_amount"))
        if v is None:
            continue
        aid = (it.get("account_id") or "").strip()
        nm = (it.get("account_nm") or "").strip().replace(" ", "")
        by_id.setdefault(aid, v)
        by_nm.setdefault(nm, v)
    for k, (aid, names) in EXTRA.items():
        val = by_id.get(aid)
        if val is None:
            val = next((by_nm[n] for n in names if n in by_nm), None)
        base[k] = val
    return base


def div(x, y):
    if x is None or y in (None, 0):
        return None
    return x / y


def compute(c, p, mcap=None):
    ta, tl = c["assets"], c["liabilities"]
    ca, cl = c["cur_assets"], c["cur_liab"]
    wc = None if ca is None or cl is None else ca - cl
    ni, ni_p = c["net_income"], (p or {}).get("net_income")
    r = {
        "Z_X1": div(wc, ta), "Z_X2": div(c["retained"], ta),
        "Z_X3": div(c["op_income"], ta), "Z_X4": div(c["equity"], tl),
        "O_SIZE": np.log(ta) if ta and ta > 0 else None,
        "O_TLTA": div(tl, ta), "O_WCTA": div(wc, ta), "O_CLCA": div(cl, ca),
        "O_OENEG": None if ta is None or tl is None else float(tl > ta),
        "O_NITA": div(ni, ta),
        # 원식: 영업자금 = 세전이익 + 감가상각비. 이인로(2015)가 영업현금흐름으로 잘못
        # 쓰는 관행을 지적했다. 감가상각비가 없으면 결측으로 두고 CFO판을 따로 둔다
        "O_FUTL": (None if c["pretax"] is None or c.get("depr") is None
                   else div(c["pretax"] + c["depr"], tl)),
        "O_FUTL_CFO": div(c["cfo"], tl),
        "O_INTWO": None if ni is None or ni_p is None else float(ni < 0 and ni_p < 0),
        "O_CHIN": (None if ni is None or ni_p is None or (abs(ni) + abs(ni_p)) == 0
                   else (ni - ni_p) / (abs(ni) + abs(ni_p))),
    }
    rev = c["revenue"]
    r["Z_X5"] = div(rev, ta)
    r["K_lnTA"] = r["O_SIZE"]
    r["K_lnSLTA"] = np.log(rev / ta) if rev and ta and rev > 0 and ta > 0 else None
    r["K_RETA"] = r["Z_X2"]
    r["K_METL"] = div(mcap, tl)
    parts = [r["K_lnTA"], r["K_lnSLTA"], r["K_RETA"], r["K_METL"]]
    r["KSCORE"] = (None if any(v is None for v in parts) else
                   -17.9 + 1.5 * parts[0] + 3.0 * parts[1] + 14.8 * parts[2] + 1.5 * parts[3])
    zx = [r["Z_X1"], r["Z_X2"], r["Z_X3"], r["K_METL"], r["Z_X5"]]
    r["ZSCORE"] = (None if any(v is None for v in zx) else
                   1.2 * zx[0] + 1.4 * zx[1] + 3.3 * zx[2] + 0.6 * zx[3] + 1.0 * zx[4])
    zs = [r["Z_X1"], r["Z_X2"], r["Z_X3"], r["Z_X4"]]
    r["Z2"] = (None if any(z is None for z in zs)
               else 6.56 * zs[0] + 3.26 * zs[1] + 6.72 * zs[2] + 1.05 * zs[3])
    o = [r[k] for k in ("O_SIZE", "O_TLTA", "O_WCTA", "O_CLCA", "O_OENEG",
                        "O_NITA", "O_FUTL", "O_INTWO", "O_CHIN")]
    r["O_SCORE"] = (None if any(v is None for v in o) else
                    -1.32 - 0.407 * o[0] + 6.03 * o[1] - 1.43 * o[2] + 0.0757 * o[3]
                    - 1.72 * o[4] - 2.37 * o[5] - 1.83 * o[6] + 0.285 * o[7] - 0.521 * o[8])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_full")
    a = ap.parse_args()
    src = pd.read_csv(FULL / f"ratios{a.suffix}.csv", dtype={"corp_code": str})
    src["corp_code"] = src["corp_code"].str.zfill(8)
    firms = sorted(src.corp_code.unique())
    years = list(range(int(src.bsns_year.min()) - 1, int(src.bsns_year.max()) + 1))

    d = Dart()
    raw = {}
    for i, code in enumerate(firms, 1):
        for fy in years:
            items = []
            for fs in ("CFS", "OFS"):
                try:
                    items = d.financials(code, fy, fs_div=fs)
                except DartError:
                    items = []
                if items:
                    break
            if items:
                raw[(code, fy)] = extract(items)
        if i % 200 == 0:
            print(f"  {i}/{len(firms)}", flush=True)

    # 시가총액 — 그 회계연도 결산월(12월 말) 스냅샷. 종목코드로 붙인다
    mc = pd.read_csv(FULL / "marketcap.csv", dtype={"ticker": str, "date": str})
    mc = mc[mc.date.str[4:6] == "12"].copy()
    mc["year"] = mc.date.str[:4].astype(int)
    mc["ticker"] = mc.ticker.str.zfill(6)
    codes = pd.DataFrame(d.corp_codes())
    codes["stock_code"] = codes.stock_code.astype(str).str.strip().str.zfill(6)
    c2s = (codes[codes.stock_code.str.len() == 6]
           .drop_duplicates("corp_code").set_index("corp_code")["stock_code"])
    mcap = {(t, y): v for t, y, v in zip(mc.ticker, mc.year, mc.mcap)}

    rows = []
    for (code, fy), c in raw.items():
        if fy == years[0]:
            continue
        r = compute(c, raw.get((code, fy - 1)), mcap.get((c2s.get(code), fy)))
        r.update(corp_code=code, bsns_year=fy)
        rows.append(r)
    df = pd.DataFrame(rows)[["corp_code", "bsns_year"] + COLS]
    df.to_csv(FULL / f"classic{a.suffix}.csv", index=False, encoding="utf-8-sig")
    print(f"\n고전 모형 변수 {len(df):,}건 / {df.corp_code.nunique():,}개사")
    print((df[COLS].isna().mean() * 100).round(1).to_string())


if __name__ == "__main__":
    main()
