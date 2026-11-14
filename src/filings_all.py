"""수시공시 수집 — 공시 행태 후보 변수의 재료.

정기보고서(A) 말고 주요사항보고서(B)·발행공시(C)·기타공시(E)·거래소공시(I)를
기업별로 받는다. 시장 구분(corp_cls)으로 월별 순회하면 상장폐지 기업이 빠지므로
(코스닥 53%·코스피 45% 누락을 이미 겪었다) 반드시 corp_code로 받는다.

출력: data/full/filings_all.csv
      (corp_code, rcept_dt, rcept_no, pblntf_ty, report_nm, flr_nm)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dart import Dart, DartError

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", default="ratios_full.csv")
    ap.add_argument("--types", nargs="+", default=["B", "C", "E", "I"])
    ap.add_argument("--bgn", default="20120101")
    ap.add_argument("--end", default="20241231")
    ap.add_argument("--out", default="filings_all.csv")
    a = ap.parse_args()

    firms = sorted(pd.read_csv(FULL / a.ratios, dtype={"corp_code": str})
                   .corp_code.str.zfill(8).unique())
    d = Dart()
    rows = []
    for i, code in enumerate(firms, 1):
        for ty in a.types:
            try:
                for x in d.list_filings(a.bgn, a.end, corp_code=code, pblntf_ty=ty):
                    rows.append({"corp_code": code, "rcept_dt": x.get("rcept_dt"),
                                 "rcept_no": x.get("rcept_no"), "pblntf_ty": ty,
                                 "report_nm": (x.get("report_nm") or "").strip(),
                                 "flr_nm": x.get("flr_nm", "")})
            except DartError:
                continue
        if i % 100 == 0:
            print(f"  {i}/{len(firms)} · 누적 {len(rows):,}건", flush=True)

    df = pd.DataFrame(rows).drop_duplicates(["rcept_no", "pblntf_ty"])
    df.to_csv(FULL / a.out, index=False, encoding="utf-8-sig")
    print(f"\n수시공시 {len(df):,}건 / {df.corp_code.nunique():,}개사 → {a.out}")
    print(df.pblntf_ty.value_counts().to_string())


if __name__ == "__main__":
    main()
