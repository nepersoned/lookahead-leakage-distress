"""3분기보고서 재무 수치 수집 — B′(접수일 정렬 + 3분기보고서) 조건의 재료.

1월 1일 예측 시점에는 직전 연도 3분기보고서(11월 중순 마감)가 이미 공개돼 있다. 본문 B는 사업보고서만 써서
'B가 쓸 수 있는 정보를 안 쓴 몫'이 누출 추정에 섞였다는 반론이 가능하다. 이를 재기 위해 3분기 수치를 모은다.

자료: DART 단일회사 전체 재무제표 API(fnlttSinglAcntAll, reprt_code=11014), 연결(CFS) 우선·없으면 별도(OFS).
    재무상태표 항목  thstrm_amount (3분기말)
    손익 항목        thstrm_add_amount (당기 누적 9개월), frmtrm_add_amount (전년 같은 기간 누적)
    현금흐름         thstrm_amount (현금흐름표는 누적으로 공시)
주의: API 값은 나중 정정이 반영된 현재값이다. 3분기 자료를 B에 더할 때 이것은 B를 유리하게(누출 추정을 작게) 만드는
      보수적 방향이다(정정 성분은 결산일 정렬에서만 컸다, final41).
접수일: data/full/reports_full.csv의 3분기보고서 최초 접수일(정정 제외). 예측 시점 이전 접수만 쓴다(분석 쪽에서 확인).
대상: results/final27_preds.csv와 같은 기업, 사업연도 2016–2023 (예측 연도 2017–2024의 직전 연도).
출력: data/full/q3_fin.csv
실행: 저장소 루트에서 `python src/q3_fin.py`  (캐시에 없는 것만 API 호출, 키 값은 출력하지 않는다)
"""
from __future__ import annotations

import sys

import pandas as pd

from dart import Dart, DartError
import vars21 as v21

ROOT_FULL = v21.FULL if hasattr(v21, "FULL") else None
YEARS = range(2016, 2024)

BS = {"assets": ("ifrs-full_Assets", ["자산총계"]),
      "liabilities": ("ifrs-full_Liabilities", ["부채총계"]),
      "equity": ("ifrs-full_Equity", ["자본총계"]),
      "cur_assets": ("ifrs-full_CurrentAssets", ["유동자산"]),
      "cur_liab": ("ifrs-full_CurrentLiabilities", ["유동부채"]),
      "retained": ("ifrs-full_RetainedEarnings", ["이익잉여금", "이익잉여금(결손금)", "결손금"])}
FLOW = {"revenue": ("ifrs-full_Revenue", ["매출액", "수익(매출액)", "영업수익", "매출"]),
        "op_income": ("dart_OperatingIncomeLoss", ["영업이익", "영업이익(손실)", "영업손실"]),
        "net_income": ("ifrs-full_ProfitLoss", ["당기순이익", "당기순이익(손실)", "분기순이익", "분기순이익(손실)",
                                                "반기순이익", "연결분기순이익"])}
CF = {"cfo": ("ifrs-full_CashFlowsFromUsedInOperatingActivities",
              ["영업활동현금흐름", "영업활동으로인한현금흐름", "영업활동으로인한순현금흐름"])}


def pick(items, spec, field, sj=None):
    by_id, by_nm = {}, {}
    for it in items:
        if sj and it.get("sj_div") not in sj:
            continue
        v = v21._num(it.get(field))
        if v is None:
            continue
        aid = (it.get("account_id") or "").strip()
        nm = (it.get("account_nm") or "").strip().replace(" ", "")
        by_id.setdefault(aid, v)
        by_nm.setdefault(nm, v)
    out = {}
    for k, (aid, names) in spec.items():
        v = by_id.get(aid)
        if v is None:
            v = next((by_nm[n] for n in names if n in by_nm), None)
        out[k] = v
    return out


def main():
    P = pd.read_csv("results/final27_preds.csv", dtype={"corp_code": str})
    firms = sorted(set(P.corp_code.str.zfill(8)))
    d = Dart()
    rows, calls = [], 0
    for i, code in enumerate(firms, 1):
        for fy in YEARS:
            items, used = [], None
            for fs in ("CFS", "OFS"):
                try:
                    items = d.financials(code, fy, reprt_code="11014", fs_div=fs)
                except DartError:
                    items = []
                if items:
                    used = fs
                    break
            if not items:
                continue
            r = {"corp_code": code, "bsns_year": fy, "fs": used, "rcept_no": items[0].get("rcept_no")}
            r.update(pick(items, BS, "thstrm_amount", {"BS"}))
            r.update({f"{k}_9m": v for k, v in pick(items, FLOW, "thstrm_add_amount", {"IS", "CIS"}).items()})
            r.update({f"{k}_9m_prev": v for k, v in pick(items, FLOW, "frmtrm_add_amount", {"IS", "CIS"}).items()})
            r.update({f"{k}_9m": v for k, v in pick(items, CF, "thstrm_amount", {"CF"}).items()})
            rows.append(r)
        if i % 100 == 0:
            print(f"  {i}/{len(firms)} 기업, 행 {len(rows):,}", flush=True)
    df = pd.DataFrame(rows)
    out = v21.FULL / "q3_fin.csv" if hasattr(v21, "FULL") else "data/full/q3_fin.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"3분기 재무 {len(df):,}행 / {df.corp_code.nunique():,}개사")
    print((df.drop(columns=["corp_code", "bsns_year", "fs", "rcept_no"]).isna().mean() * 100).round(1).to_string())


if __name__ == "__main__":
    sys.exit(main())
