"""2025년 홀드아웃 자료 준비 — FY2024 사업보고서를 기존 파이프라인 그대로 1년 늘린다.

사전 계획: results/holdout2025_plan.md. 채점은 하지 않는다(자료 준비·점검만).

단계 (기존 코드를 그대로 부른다)
    1. 정기보고서 목록  gate3a_reports 방식. 2025-01-01~12-31 접수분, 코스닥 시장 순회
                        + 현재 시장 구분을 잃은 기업(상장폐지·이전상장)은 기업별 보충(collect_firms)
                        → data/full/reports_full_2025.csv
    2. 원문 ZIP         t1_parse.fetch (document.xml) → data/cache/docs/{rcept_no}.zip
                        대상: FY2024 사업보고서 최초 제출본(정정 아님), 기업당 가장 이른 것
    3. 파싱             orig_fin2.parse → data/full/orig_fin2_2025.csv
    4. classic 15개     final23.build_classic (기존 orig_fin2 + 새 파일, FY2024만 저장)
                        영업현금흐름은 vars_classic 방식으로 FY2024 API만 받는다
                        → data/full/classic_orig_fy2024.csv

사용: python src/holdout2025_data.py [--stage all|reports|docs|parse|classic]
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate3a_reports as g3a  # noqa: E402
import orig_fin as of  # noqa: E402
import orig_fin2 as of2  # noqa: E402
import t1_parse as t1  # noqa: E402
from dart import Dart, DartError  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"
DOCS = ROOT / "data" / "cache" / "docs"
REP_OUT = FULL / "reports_full_2025.csv"
FIN_OUT = FULL / "orig_fin2_2025.csv"
CLS_OUT = FULL / "classic_orig_fy2024.csv"
API_OUT = FULL / "classic_api_fy2024.csv"
FAIL_OUT = FULL / "orig_fin2_2025_status.csv"

load_dotenv(ROOT / ".env")
_KEY = os.getenv("DART_API_KEY", "").strip()


def safe(msg) -> str:
    """예외 문자열에 URL(인증키 포함)이 섞일 수 있다. 키를 지운다."""
    s = str(msg)
    return s.replace(_KEY, "***") if _KEY else s


# ---------- 1. 정기보고서 목록 ----------
def sweep_2025() -> pd.DataFrame:
    """gate3a_reports.collect와 같은 규칙. 단, 접수월을 2025년 1~12월로 고정하고
    그 기간에 접수된 FY2024·FY2025 보고서를 모두 받는다."""
    dart = Dart()
    rows = []
    for m in pd.date_range("2025-01-01", "2025-12-01", freq="MS"):
        bgn = m.strftime("%Y%m%d")
        end = (m + pd.offsets.MonthEnd(1)).strftime("%Y%m%d")
        print(f"  정기공시 검색: 제출월 {bgn[:6]}", flush=True)
        for f in dart.list_filings(bgn, end, corp_cls="K", pblntf_ty="A"):
            nm = f.get("report_nm", "")
            m2 = g3a.RE_REPORT.search(nm)
            if not m2:
                continue
            rtype, year, month = m2.group(1), int(m2.group(2)), int(m2.group(3))
            if year not in (2024, 2025) or month not in g3a.PERIOD_MONTHS[rtype]:
                continue
            rows.append({
                "corp_code": f.get("corp_code"), "corp_name": f.get("corp_name"),
                "stock_code": f.get("stock_code"), "corp_cls": f.get("corp_cls"),
                "report_type": rtype, "bsns_year": year, "period_end": month,
                "rcept_dt": f.get("rcept_dt"), "rcept_no": f.get("rcept_no"),
                "is_amended": nm.startswith("[기재정정]") or "정정" in nm,
                "report_nm": nm,
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["rcept_no"])
    df["rcept_dt"] = pd.to_datetime(df["rcept_dt"], format="%Y%m%d")
    return df


def stage_reports():
    if REP_OUT.exists():
        print(f"[1] 이미 있음: {REP_OUT.name} — 건너뜀")
        return
    k = sweep_2025()
    print(f"  시장 순회 {len(k):,}건")
    # 보충 대상: 기존 목록에서 FY2023·FY2024 보고서가 있는 기업 중 FY2024 사업보고서가 안 잡힌 곳
    old = pd.read_csv(FULL / "reports_full.csv", dtype={"corp_code": str})
    old["corp_code"] = old.corp_code.str.zfill(8)
    cand = set(old.loc[old.bsns_year.isin([2023, 2024]), "corp_code"])
    got = set(k.loc[(k.report_type == "사업") & (k.bsns_year == 2024), "corp_code"])
    miss = sorted(cand - got)
    print(f"  기업별 보충 대상 {len(miss):,}곳 (시장 순회에서 FY2024 사업보고서 없음)", flush=True)
    sup = g3a.collect_firms(miss, "20250101", "20251231")
    if not sup.empty:
        sup = sup[sup.bsns_year.isin([2024, 2025])]
        print(f"  보충 {len(sup):,}건", flush=True)
    df = pd.concat([k, sup]).drop_duplicates(subset=["rcept_no"])
    df["corp_code"] = df.corp_code.astype(str).str.zfill(8)
    df = df.sort_values(["corp_code", "bsns_year", "period_end", "rcept_dt"]).reset_index(drop=True)
    df.to_csv(REP_OUT, index=False, encoding="utf-8-sig")
    print(f"[1] {len(df):,}건 → {REP_OUT}")
    print(df.groupby(["bsns_year", "report_type"]).size().to_string())


def targets() -> pd.DataFrame:
    """FY2024 사업보고서 최초 제출본 — 기업당 가장 이른 비정정 접수 (index.csv 규칙과 같다)."""
    r = pd.read_csv(REP_OUT, dtype={"corp_code": str, "rcept_no": str})
    r["corp_code"] = r.corp_code.str.zfill(8)
    r["is_amended"] = r.is_amended.astype(str).str.lower().isin(["true", "1"])
    a = r[(r.report_type == "사업") & (r.bsns_year == 2024) & (~r.is_amended)]
    return a.sort_values(["rcept_dt", "rcept_no"]).drop_duplicates("corp_code").reset_index(drop=True)


# ---------- 2. 원문 ZIP ----------
def stage_docs():
    t = targets()
    todo = [rc for rc in t.rcept_no if not (DOCS / f"{rc}.zip").exists()]
    print(f"[2] 대상 {len(t):,}건 / 캐시 있음 {len(t) - len(todo):,} / 받을 것 {len(todo):,}", flush=True)
    fail = []
    for i, rc in enumerate(todo, 1):
        try:
            t1.fetch(rc, _KEY)
        except Exception as exc:  # noqa: BLE001
            fail.append(rc)
            print(f"  실패 {rc}: {type(exc).__name__} {safe(exc)[:150]}", flush=True)
        if i % 100 == 0:
            print(f"  {i}/{len(todo)} 실패 {len(fail)}", flush=True)
    print(f"[2] 완료 — 다운로드 실패 {len(fail)}건", flush=True)


# ---------- 3. 파싱 ----------
def stage_parse():
    t = targets()
    rows, status = [], []
    for rc, code, name in zip(t.rcept_no, t.corp_code, t.corp_name):
        p = DOCS / f"{rc}.zip"
        xml = of.read_xml(p) if p.exists() else None
        got = of2.parse(xml) if xml else {}
        st = "ZIP없음" if not p.exists() else ("읽기실패" if not xml else
                                               ("성공" if got else "파싱실패"))
        status.append({"corp_code": code, "corp_name": name, "rcept_no": rc,
                       "status": st, "has_fy2024": 2024 in got})
        for y, v in got.items():
            rows.append({"corp_code": code, "bsns_year": y, "src_rcept": rc,
                         "src_fy": 2024, **v})
    df = (pd.DataFrame(rows).sort_values("src_rcept")
          .drop_duplicates(["corp_code", "bsns_year"]))
    # 기존 orig_fin2.csv와 같은 열 순서
    cols = list(pd.read_csv(FULL / "orig_fin2.csv", nrows=0).columns)
    df = df.reindex(columns=cols + [c for c in df.columns if c not in cols])
    df.to_csv(FIN_OUT, index=False, encoding="utf-8-sig")
    st = pd.DataFrame(status)
    st.to_csv(FAIL_OUT, index=False, encoding="utf-8-sig")
    n_fail = int((st.status != "성공").sum())
    print(f"[3] 문서 {len(st):,}건 / 파싱 실패 {n_fail}건 ({n_fail/len(st):.1%})")
    print(st.status.value_counts().to_string())
    print(f"  추출 {len(df):,}행 / FY2024 행 {int((df.bsns_year == 2024).sum()):,} → {FIN_OUT}")
    print((df[df.bsns_year == 2024].drop(columns=["corp_code", "src_rcept"]).isna().mean() * 100)
          .round(1).to_string())


# ---------- 4. classic ----------
def stage_classic():
    import vars_classic as vc
    from final23 import build_classic

    t = targets()
    d = Dart()
    raw = {}
    for i, code in enumerate(t.corp_code, 1):
        for fy in (2023, 2024):
            items = []
            for fs in ("CFS", "OFS"):
                try:
                    items = d.financials(code, fy, fs_div=fs)
                except DartError as exc:
                    print(f"  API {code} FY{fy} {fs}: {safe(exc)[:80]}", flush=True)
                    items = []
                except Exception as exc:  # noqa: BLE001
                    print(f"  API {code} FY{fy} {fs}: {type(exc).__name__} {safe(exc)[:120]}",
                          flush=True)
                    items = []
                if items:
                    break
            if items:
                raw[(code, fy)] = vc.extract(items)
        if i % 200 == 0:
            print(f"  API {i}/{len(t)}", flush=True)

    # 시가총액 — 결산월(2024년 12월 말) 스냅샷. vars_classic·final23과 같은 규칙
    mc = pd.read_csv(FULL / "marketcap.csv", dtype={"ticker": str, "date": str})
    mc = mc[mc.date.str[4:6] == "12"].copy()
    mc["year"] = mc.date.str[:4].astype(int)
    mc["ticker"] = mc.ticker.str.zfill(6)
    mcap = {(tk, y): v for tk, y, v in zip(mc.ticker, mc.year, mc.mcap)}
    codes = pd.DataFrame(d.corp_codes())
    codes["stock_code"] = codes.stock_code.astype(str).str.strip().str.zfill(6)
    c2s = (codes[codes.stock_code.str.len() == 6]
           .drop_duplicates("corp_code").set_index("corp_code")["stock_code"])

    api_rows = []
    for code in t.corp_code:
        c = raw.get((code, 2024))
        if c is None:
            continue
        r = vc.compute(c, raw.get((code, 2023)), mcap.get((c2s.get(code), 2024)))
        r.update(corp_code=code, bsns_year=2024)
        api_rows.append(r)
    api = pd.DataFrame(api_rows)[["corp_code", "bsns_year"] + vc.COLS]
    api.to_csv(API_OUT, index=False, encoding="utf-8-sig")
    print(f"[4] API판 FY2024 {len(api):,}행 → {API_OUT}")

    # build_classic은 classic_full(API판)의 O_FUTL_CFO를 쓴다. 기존 + FY2024를 합친다
    old_api = pd.read_csv(FULL / "classic_full.csv", dtype={"corp_code": str})
    old_api["corp_code"] = old_api.corp_code.str.zfill(8)
    api_all = pd.concat([old_api, api]).drop_duplicates(["corp_code", "bsns_year"])
    doc_old = pd.read_csv(FULL / "orig_fin2.csv", dtype={"corp_code": str})
    doc_new = pd.read_csv(FIN_OUT, dtype={"corp_code": str})
    doc = (pd.concat([doc_old, doc_new]).assign(corp_code=lambda x: x.corp_code.str.zfill(8))
           .sort_values("src_rcept").drop_duplicates(["corp_code", "bsns_year"]))
    cls = build_classic(doc, api_all, mcap, c2s)
    cls = cls[cls.bsns_year == 2024].reset_index(drop=True)
    cols = list(pd.read_csv(FULL / "classic_orig.csv", nrows=0).columns)
    cls = cls[cols]
    cls.to_csv(CLS_OUT, index=False, encoding="utf-8-sig")
    print(f"[4] classic 원문판 FY2024 {len(cls):,}행 → {CLS_OUT}")
    print((cls[cols[2:]].isna().mean() * 100).round(1).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "reports", "docs", "parse", "classic"])
    a = ap.parse_args()
    stages = ["reports", "docs", "parse", "classic"] if a.stage == "all" else [a.stage]
    for s in stages:
        globals()[f"stage_{s}"]()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        print(safe(traceback.format_exc()), flush=True)
        sys.exit(1)
