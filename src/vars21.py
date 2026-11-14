"""변수 세트 2 — 윤양현 외(2022) 명세의 재무비율 21개.

논문 <표 7>의 산출식을 그대로 옮긴다. 출처 계보는
김일(2005) → 박창래·서영미(2015) → 신동인·곽기영(2018) → 윤양현 외(2022).

우리 변수 10개(gate3b)와 겹치지 않는 별도 세트다. 목적은 복제가 아니라
**"변수를 바꿔도 A−B가 남는가"**를 보이는 것이다.

논문과 다를 수밖에 없는 점 두 가지 — 결과에 그대로 적는다.
  1. 자료원이 FnGuide가 아니라 DART다. 계정 매핑이 완전히 같지 않다.
  2. '총차입부채'는 IFRS 표준계정이 없다. 차입금·사채를 합해 근사한다.

출력: data/full/accounts_full.csv (원계정), data/full/vars21_full.csv (비율)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dart import Dart, DartError

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"

# 계정 → (account_id, 이름 후보). 이름은 공백 제거 후 대조한다.
ACC = {
    # 재무상태표
    "assets":      ("ifrs-full_Assets", ["자산총계"]),
    "liabilities": ("ifrs-full_Liabilities", ["부채총계"]),
    "equity":      ("ifrs-full_Equity", ["자본총계"]),
    "equity_parent": ("ifrs-full_EquityAttributableToOwnersOfParent",
                      ["지배기업의소유주에게귀속되는자본", "지배기업소유주지분",
                       "지배기업의소유주지분"]),
    "capital":     ("ifrs-full_IssuedCapital", ["자본금"]),
    # 손익
    "revenue":     ("ifrs-full_Revenue", ["매출액", "수익(매출액)", "영업수익"]),
    "op_income":   ("dart_OperatingIncomeLoss", ["영업이익", "영업이익(손실)"]),
    "pretax":      ("ifrs-full_ProfitLossBeforeTax",
                    ["법인세비용차감전순이익(손실)", "법인세비용차감전순이익",
                     "법인세비용차감전계속사업이익(손실)"]),
    "net_income":  ("ifrs-full_ProfitLoss", ["당기순이익", "당기순이익(손실)"]),
    "compr":       ("ifrs-full_ComprehensiveIncome", ["총포괄손익", "총포괄이익"]),
    # 현금흐름 — 논문의 '현금흐름'은 영업활동현금흐름으로 본다
    "cfo":         ("ifrs-full_CashFlowsFromUsedInOperatingActivities",
                    ["영업활동현금흐름", "영업활동으로인한현금흐름",
                     "영업활동으로인한순현금흐름"]),
}

# 총차입부채 근사에 쓰는 계정들 (있는 것만 더한다)
DEBT_NAMES = ["단기차입금", "장기차입금", "유동성장기차입금", "사채",
              "유동성사채", "단기사채", "전환사채", "신주인수권부사채"]

VAR_COLS = [f"P{i}" for i in range(1, 8)] + [f"S{i}" for i in range(1, 8)] + \
           [f"A{i}" for i in range(1, 5)] + [f"G{i}" for i in range(1, 4)]


def _num(s):
    if not s or str(s).strip() in ("-", "", "nan"):
        return None
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def extract(items: list[dict]) -> dict:
    by_id, by_nm = {}, {}
    for it in items:
        v = _num(it.get("thstrm_amount"))
        if v is None:
            continue
        aid = (it.get("account_id") or "").strip()
        nm = (it.get("account_nm") or "").strip().replace(" ", "")
        if aid and aid != "-dart_None" and aid not in by_id:
            by_id[aid] = v
        if nm and nm not in by_nm:
            by_nm[nm] = v

    out = {}
    for key, (aid, names) in ACC.items():
        v = by_id.get(aid)
        if v is None:
            for n in names:
                v = by_nm.get(n.replace(" ", ""))
                if v is not None:
                    break
        out[key] = v

    # 총차입부채 — 표준계정이 없어 이름으로 긁어 더한다
    debt = [by_nm[n] for n in DEBT_NAMES if n in by_nm]
    out["debt_total"] = sum(debt) if debt else None

    # 비지배지분이 없는 기업은 지배주주지분 = 자본총계다. 개별재무제표도 같다.
    if out["equity_parent"] is None:
        out["equity_parent"] = out["equity"]
    return out


def _avg(cur, prv):
    """논문의 '(평균)'은 기초·기말 평균이다. 전기가 없으면 당기값으로 대체한다."""
    if cur is None:
        return None
    return cur if prv is None else (cur + prv) / 2


def compute(cur: dict, prv: dict | None) -> dict:
    prv = prv or {}

    def d(x, y, pct=True):
        if x is None or y is None or y == 0:
            return None
        return x / y * (100 if pct else 1)

    a_assets = _avg(cur["assets"], prv.get("assets"))
    a_equity = _avg(cur["equity"], prv.get("equity"))
    a_parent = _avg(cur["equity_parent"], prv.get("equity_parent"))
    a_liab = _avg(cur["liabilities"], prv.get("liabilities"))
    a_capital = _avg(cur["capital"], prv.get("capital"))
    a_revenue = _avg(cur["revenue"], prv.get("revenue"))
    a_debt = _avg(cur["debt_total"], prv.get("debt_total"))
    cfo, rev = cur["cfo"], cur["revenue"]

    v = {
        # 수익성 — ROA/ROE 계열
        "P1": d(cur["op_income"], a_assets),
        "P2": d(cur["pretax"], a_assets),
        "P3": d(cur["net_income"], a_assets),
        "P4": d(cur["compr"], a_assets),
        "P5": d(cur["op_income"], a_equity),
        "P6": d(cur["pretax"], a_equity),
        "P7": d(cur["net_income"], a_equity),
        # 안정성
        "S1": d(cur["liabilities"], cur["equity"]),
        "S2": d(cfo, a_parent),
        "S3": d(cfo, a_revenue),
        "S4": d(cfo, a_liab),
        "S5": d(cfo, a_equity),
        "S6": d(cfo, a_assets),
        "S7": d(cfo, a_debt),
        # 활동성 (회전율은 배수라 100을 곱하지 않는다)
        "A1": d(rev, a_assets, pct=False),
        "A2": d(rev, a_capital, pct=False),
        "A3": d(rev, a_liab, pct=False),
        "A4": d(rev, a_equity, pct=False),
        # 성장성 — 전기가 없으면 만들 수 없다
        "G1": None, "G2": None, "G3": None,
    }
    for key, fld in (("G1", "liabilities"), ("G2", "assets"), ("G3", "revenue")):
        c, p = cur.get(fld), prv.get(fld)
        if c is not None and p not in (None, 0):
            v[key] = (c / p - 1) * 100
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reprt-code", default="11011")
    ap.add_argument("--suffix", default="_full")
    ap.add_argument("--limit", type=int, default=0, help="시험 실행용 기업 수")
    a = ap.parse_args()

    ratios = pd.read_csv(FULL / f"ratios{a.suffix}.csv", dtype={"corp_code": str})
    ratios["corp_code"] = ratios["corp_code"].str.zfill(8)
    firms = sorted(ratios["corp_code"].unique())
    if a.limit:
        firms = firms[:a.limit]
    years = sorted(ratios["bsns_year"].unique())
    # 성장률·평균에 전기가 필요하므로 한 해 앞까지 받는다
    years = list(range(min(years) - 1, max(years) + 1))

    dart = Dart()
    raw: dict[tuple[str, int], dict] = {}
    miss = 0
    for i, code in enumerate(firms, 1):
        for fy in years:
            items = []
            for fs in ("CFS", "OFS"):
                try:
                    items = dart.financials(code, fy, reprt_code=a.reprt_code,
                                            fs_div=fs)
                except DartError:
                    items = []
                if items:
                    break
            if items:
                raw[(code, fy)] = extract(items)
            else:
                miss += 1
        if i % 100 == 0:
            print(f"  {i}/{len(firms)}개사", flush=True)

    rows = []
    for (code, fy), cur in raw.items():
        if fy == min(years):      # 전기 전용으로만 받은 해
            continue
        v = compute(cur, raw.get((code, fy - 1)))
        v.update({"corp_code": code, "bsns_year": fy})
        rows.append(v)

    df = pd.DataFrame(rows)[["corp_code", "bsns_year"] + VAR_COLS]
    df.to_csv(FULL / f"vars21{a.suffix}.csv", index=False, encoding="utf-8-sig")
    print(f"\n변수 21개 {len(df):,}건 / {df.corp_code.nunique():,}개사 "
          f"→ vars21{a.suffix}.csv")
    print(f"조회 실패 {miss:,}건")
    print("\n[변수별 결측률 %]")
    print((df[VAR_COLS].isna().mean() * 100).round(1).to_string())


if __name__ == "__main__":
    main()
