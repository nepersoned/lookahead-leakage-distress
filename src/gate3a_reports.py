"""관문 3-a — 정기보고서 공시 인덱스 수집 (이 연구의 심장: rcept_dt).

각 기업×보고서가 **실제로 세상에 공개된 날짜**를 모은다.
실험 A와 B를 가르는 유일한 필드.

사업보고서만이 아니라 분기·반기보고서도 받는다. 실험 B가 사업보고서만 보면
T=2월에 1년 묵은 자료를 쓰게 되는데, 실제로 그 시점엔 직전 3분기보고서가
공개되어 있기 때문이다(11월 제출). 이걸 받아야 B'(분기 포함 정합 조건)를
만들 수 있다.

부수 효과: KOSDAQ 상장기업 모집단(사업보고서를 낸 기업)도 여기서 나온다.
현재 상장사 목록에서 뽑지 않으므로 이미 퇴출된 기업도 그해 보고서가 있으면 남는다.

출력: <out>/reports_full.csv
      (corp_code, report_type, bsns_year, period_end, rcept_dt, is_amended)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from dart import Dart

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw"

# 예: "사업보고서 (2022.12)" / "[기재정정]반기보고서 (2022.06)" / "분기보고서 (2022.09)"
RE_REPORT = re.compile(r"(사업|반기|분기)보고서\s*\((\d{4})\.(\d{2})\)")

# 보고서 종류 → 12월 결산 기준으로 기대되는 기준월
PERIOD_MONTHS = {"사업": {12}, "반기": {6}, "분기": {3, 9}}


def collect(years: list[int], corp_cls: str = "K",
            types: list[str] | None = None) -> pd.DataFrame:
    """정기공시(pblntf_ty=A)를 제출월 단위로 훑어 보고서 인덱스를 만든다.

    years : 사업연도 목록
    types : ["사업", "반기", "분기"] 중 받을 것. 기본은 전부
    """
    types = types or ["사업", "반기", "분기"]
    dart = Dart()
    rows = []
    for fy in years:
        # 분기·반기는 해당 연도 안에, 사업보고서는 다음 해에 제출된다.
        # 정정까지 잡으려면 fy년 1월 ~ fy+1년 12월을 훑는다.
        for m in pd.date_range(f"{fy}-01-01", f"{fy+1}-12-01", freq="MS"):
            bgn = m.strftime("%Y%m%d")
            end = (m + pd.offsets.MonthEnd(1)).strftime("%Y%m%d")
            print(f"  정기공시 검색: FY{fy} 제출월 {bgn[:6]}", flush=True)
            for f in dart.list_filings(bgn, end, corp_cls=corp_cls, pblntf_ty="A"):
                nm = f.get("report_nm", "")
                m2 = RE_REPORT.search(nm)
                if not m2:
                    continue
                rtype, year, month = m2.group(1), int(m2.group(2)), int(m2.group(3))
                if rtype not in types or year != fy:
                    continue
                if month not in PERIOD_MONTHS[rtype]:  # 12월 결산법인만
                    continue
                rows.append(
                    {
                        "corp_code": f.get("corp_code"),
                        "corp_name": f.get("corp_name"),
                        "stock_code": f.get("stock_code"),
                        "corp_cls": f.get("corp_cls"),
                        "report_type": rtype,
                        "bsns_year": fy,
                        "period_end": month,
                        "rcept_dt": f.get("rcept_dt"),
                        "rcept_no": f.get("rcept_no"),
                        "is_amended": nm.startswith("[기재정정]") or "정정" in nm,
                        "report_nm": nm,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["rcept_no"])
    df["rcept_dt"] = pd.to_datetime(df["rcept_dt"], format="%Y%m%d")
    return df.sort_values(
        ["corp_code", "bsns_year", "period_end", "rcept_dt"]
    ).reset_index(drop=True)


def collect_firms(codes, bgn: str, end: str,
                  types: list[str] | None = None) -> pd.DataFrame:
    """지정한 기업들의 정기보고서를 기업별로 직접 받는다.

    왜 필요한가 — list.json의 corp_cls 필터는 **현재** 시장 구분으로 거른다.
    상장폐지된 기업은 KOSDAQ 분류를 잃어서 과거 공시까지 통째로 빠진다.
    하필 그 기업들이 우리가 맞히려는 사건 표본이므로, 시장 단위 순회만으로는
    생존편향이 그대로 남는다. 사건기업은 corp_code로 직접 받아 메운다.
    """
    types = types or ["사업", "반기", "분기"]
    dart = Dart()
    codes = list(dict.fromkeys(str(c).zfill(8) for c in codes))
    rows = []
    for i, code in enumerate(codes, 1):
        if i % 50 == 0:
            print(f"  기업별 보충 {i}/{len(codes)}", flush=True)
        for f in dart.list_filings(bgn, end, corp_code=code, pblntf_ty="A"):
            nm = f.get("report_nm", "")
            m = RE_REPORT.search(nm)
            if not m:
                continue
            rtype, year, month = m.group(1), int(m.group(2)), int(m.group(3))
            if rtype not in types or month not in PERIOD_MONTHS[rtype]:
                continue
            rows.append({
                "corp_code": f.get("corp_code"),
                "corp_name": f.get("corp_name"),
                "stock_code": f.get("stock_code"),
                "corp_cls": f.get("corp_cls"),
                "report_type": rtype,
                "bsns_year": year,
                "period_end": month,
                "rcept_dt": f.get("rcept_dt"),
                "rcept_no": f.get("rcept_no"),
                "is_amended": nm.startswith("[기재정정]") or "정정" in nm,
                "report_nm": nm,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["rcept_no"])
    df["rcept_dt"] = pd.to_datetime(df["rcept_dt"], format="%Y%m%d")
    return df.reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", type=int,
                   default=[2018, 2019, 2020, 2021, 2022, 2023])
    p.add_argument("--corp-cls", default="K", help="Y=KOSPI K=KOSDAQ")
    p.add_argument("--types", nargs="+", default=["사업", "반기", "분기"])
    p.add_argument("--out", default=str(OUT),
                   help="출력 디렉터리. 파일럿 CSV를 덮지 않으려면 data/full")
    p.add_argument("--name", default="reports_full.csv", help="출력 파일명")
    a = p.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect(a.years, corp_cls=a.corp_cls, types=a.types)
    if df.empty:
        print("수집 0건")
        return
    df.to_csv(out_dir / a.name, index=False, encoding="utf-8-sig")

    # 원 제출(최초 공시)만 남긴 공개일 테이블 — D3 규칙: 원 제출일이 '공개일'
    orig = (df[~df["is_amended"]]
            .sort_values("rcept_dt")
            .groupby(["corp_code", "bsns_year", "period_end"], as_index=False)
            .first())

    print(f"\n총 {len(df):,}건 (원 제출 {len(orig):,}건) → {out_dir / a.name}")
    print("\n[보고서 종류별]")
    print(df["report_type"].value_counts().to_string())
    print("\n[사업연도별 원 제출 건수]")
    print(orig["bsns_year"].value_counts().sort_index().to_string())
    print("\n[원 제출 공개일의 월별 분포]  ← 3월 말 집중이 정상")
    print(orig["rcept_dt"].dt.month.value_counts().sort_index().to_string())
    print("\n[공개일 분산 — 같은 FY 안에서도 기업마다 다르다는 증거]")
    for fy, g in orig[orig["report_type"] == "사업"].groupby("bsns_year"):
        print(f"  FY{fy}: 최초 {g.rcept_dt.min().date()} / 중앙 "
              f"{g.rcept_dt.median().date()} / 최종 {g.rcept_dt.max().date()}")
    print(f"\n정정 사업보고서 비율: {df['is_amended'].mean():.1%}")


if __name__ == "__main__":
    main()
