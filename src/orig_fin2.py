"""최초 공시본 재무 파서 v2 — 요약표가 아니라 **본표**를 읽는다.

v1(orig_fin.py)은 「III. 재무에 관한 사항」의 `1. 요약재무정보` 표를 읽어 성공률이 60%에
머물렀다. 요약표는 회사마다 형식이 다르고 연도 표기가 빠지기도 한다.

v2는 같은 절의 **재무상태표·손익계산서 본표**를 읽는다. 이 표는 XBRL에서 자동 생성돼
형식이 일정하다.

    제목 표   "연결 재무상태표 / 제 24 기 2016.12.31 현재 / ..."  ← 기수와 연도가 여기 있다
    본표      "  | 제 24 기 | 제 23 기 | 제 22 기"               ← 헤더는 기수만
              "자산총계 | 88,070,435,601 | ..."                  ← 단위는 원

따라서 **제목 표에서 기수→연도를 읽고, 바로 뒤 본표의 열 순서에 맞춰** 값을 가져온다.
연결(CFS)을 먼저 쓰고 없으면 별도(OFS)를 쓴다.

사용: python src/orig_fin2.py [--limit 200] [--check]
출력: data/full/orig_fin2.csv
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import orig_fin as of

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"

CELL = re.compile(r"<T[DEH][^>]*>.*?</T[DEH]>", re.S)
ROW = re.compile(r"<TR[^>]*>.*?</TR>", re.S)
TABLE = re.compile(r"<TABLE.*?</TABLE>", re.S)
# "제 24 기", "제24기", "제 24 기 3분기" 등
GI = re.compile(r"제\s*(\d{1,3})\s*기")
YMD = re.compile(r"(20\d{2})[.\-/년]\s*(\d{1,2})")
UNIT = re.compile(r"단위\s*:?\s*(원|천원|백만원|십억원|억원)")

# 재무상태표 / 손익계산서 항목. 앞에 오는 것이 우선
BS = [
    ("assets", r"^자산총계$|^자산\s*총계$|^총자산$"),
    ("cur_assets", r"유동자산$"),
    ("liabilities", r"^부채총계$|^부채\s*총계$|^총부채$"),
    ("cur_liab", r"유동부채$"),
    ("equity", r"^자본총계$|^자본\s*총계$|지배기업.*소유주지분$"),
    ("retained", r"이익잉여금|결손금"),
]
IS_ = [
    ("revenue", r"^(매출액|매출|수익|수익\(매출액\)|영업수익|매출액및지분법손익|영업수익\(매출액\))$"),
    ("op_income", r"^영업이익|^영업손실|영업이익\(손실\)$"),
    ("pretax", r"법인세.*차감전"),
    ("net_income", r"^(당기순이익|당기순손실|당기순이익\(손실\)|연결당기순이익)"),
]


def cells_of(tab: str):
    return [[of._text(c) for c in CELL.findall(r)] for r in ROW.findall(tab)]


def caption_years(cells) -> dict[int, int]:
    """제목 표 → {기수: 연도}"""
    out = {}
    for row in cells:
        txt = " ".join(row)
        g, y = GI.search(txt), YMD.search(txt)
        if g and y:
            out[int(g.group(1))] = int(y.group(1))
    return out


def header_gi(cells) -> list[int]:
    """본표 헤더 줄에서 기수 순서를 읽는다."""
    for row in cells[:3]:
        gis = [int(GI.search(c).group(1)) for c in row if GI.search(c)]
        if len(gis) >= 2:
            return gis
    return []


def scan(xml: str, fields, label: str) -> dict[int, dict]:
    """label('재무상태표'/'손익계산서')이 든 제목 표 뒤의 본표를 읽는다."""
    out: dict[int, dict] = {}
    tabs = TABLE.findall(xml)
    scale = 1.0
    for i, tab in enumerate(tabs):
        flat = of._text(tab)
        u = UNIT.search(flat)
        if u:
            scale = SCALEMAP[u.group(1)]
        if label not in flat or "제" not in flat:
            continue
        cap = caption_years(cells_of(tab))
        if len(cap) < 2:
            continue
        # 연결을 우선하되, 이미 채웠으면 건너뛴다
        for tab2 in tabs[i + 1:i + 4]:
            flat2 = of._text(tab2)
            u2 = UNIT.search(flat2)
            sc = SCALEMAP[u2.group(1)] if u2 else scale
            cells = cells_of(tab2)
            if len(cells) < 6:
                continue
            gis = header_gi(cells)
            if not gis:
                continue
            years = [cap.get(g) for g in gis]
            if sum(y is not None for y in years) < 2:
                continue
            vals: dict[int, dict] = {y: {} for y in years if y}
            for row in cells:
                if len(row) < 2:
                    continue
                # 공백과 주석 참조만 지운다. 끝 괄호를 지우면 '수익(매출액)'이 깨진다
                lab = re.sub(r"\s|\(\s*주\s*[\d,\s]*\)", "", row[0])
                lab = re.sub(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ0-9.\-·]+", "", lab).strip()
                nums = [v for v in (of._num(c) for c in row[1:]) if v is not None]
                if len(nums) < 2:
                    continue
                for key, pat in fields:
                    if re.search(pat, lab):
                        for y, v in zip(years, nums[:len(years)]):
                            if y:
                                vals[y].setdefault(key, v * sc)
                        break
            if any(len(v) >= 2 for v in vals.values()):
                for y, v in vals.items():
                    if v:
                        out.setdefault(y, {}).update(
                            {k: x for k, x in v.items() if k not in out.get(y, {})})
                break
        if out:
            break
    return out


SCALEMAP = {"원": 1.0, "천원": 1e3, "백만원": 1e6, "억원": 1e8, "십억원": 1e9}


def slice_fin(xml: str) -> str:
    """재무제표 구간만 잘라 낸다 — 전체를 훑으면 문서당 수 초가 걸린다."""
    i = xml.find("재무상태표")
    if i < 0:
        return xml
    return xml[max(0, i - 3000):i + 700_000]


def parse(xml: str) -> dict[int, dict]:
    seg = slice_fin(xml)
    bs = scan(seg, BS, "재무상태표")
    inc = scan(seg, IS_, "손익계산서")
    out: dict[int, dict] = {}
    for y in set(bs) | set(inc):
        d = {**bs.get(y, {}), **inc.get(y, {})}
        if "assets" in d and len(d) >= 3:
            out[y] = d
    if out:
        return out
    return of.parse(xml)          # 본표를 못 읽으면 v1(요약표)로 되돌린다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    idx = pd.read_csv(ROOT / "data" / "text_zip" / "index.csv",
                      dtype={"rcept_no": str, "corp_code": str})
    idx["corp_code"] = idx.corp_code.str.zfill(8)
    if a.limit:
        idx = idx.sample(a.limit, random_state=42)

    rows, fail = [], 0
    for n, (rc, code, fy) in enumerate(zip(idx.rcept_no, idx.corp_code, idx.bsns_year), 1):
        xml = of.read_xml(of.DOCS / f"{rc}.zip")
        got = parse(xml) if xml else {}
        if not got:
            fail += 1
        for y, v in got.items():
            rows.append({"corp_code": code, "bsns_year": y, "src_rcept": rc,
                         "src_fy": int(fy), **v})
        if n % 500 == 0:
            print(f"  {n}/{len(idx)} 실패 {fail} ({fail/n:.1%})", flush=True)

    df = pd.DataFrame(rows).sort_values("src_rcept").drop_duplicates(["corp_code", "bsns_year"])
    print(f"\n문서 {len(idx):,}건 / 파싱 실패 {fail}건 ({fail/len(idx):.1%})")
    print(f"추출 {len(df):,}행 / {df.corp_code.nunique():,}개사 "
          f"{int(df.bsns_year.min())}~{int(df.bsns_year.max())}")
    print((df.drop(columns=["corp_code", "src_rcept"]).isna().mean() * 100).round(1).to_string())
    out = FULL / (f"orig_fin2{'_sample' if a.limit else ''}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    if a.check:
        api = pd.read_csv(FULL / "ratios_full.csv", dtype={"corp_code": str})
        api["corp_code"] = api.corp_code.str.zfill(8)
        rep = pd.read_csv(FULL / "reports_full.csv", dtype={"corp_code": str})
        rep["corp_code"] = rep.corp_code.str.zfill(8)
        amd = rep[(rep.report_type == "사업") &
                  rep.is_amended.astype(str).str.lower().isin(["true", "1"])]
        amended = set(zip(amd.corp_code, amd.bsns_year))
        d2 = df.assign(roa=df.net_income / df.assets,
                       asset_turnover=df.revenue / df.assets,
                       retained_to_ta=df.retained / df.assets,
                       debt_ratio=df.liabilities / (df.equity.replace(0, np.nan)))
        cols = ["roa", "asset_turnover", "retained_to_ta", "debt_ratio"]
        m = d2.merge(api[["corp_code", "bsns_year"] + cols], on=["corp_code", "bsns_year"],
                     suffixes=("_doc", "_api"))
        m["정정있음"] = [tuple(k) in amended for k in zip(m.corp_code, m.bsns_year)]
        print("\n=== 원문 파싱값 vs API값 (1% 이내 일치 비율) ===")
        for grp, d in m.groupby("정정있음"):
            line = {("정정 있음" if grp else "정정 없음"): len(d)}
            for c in cols:
                x, yv = d[f"{c}_doc"], d[f"{c}_api"]
                ok = (x - yv).abs() <= (yv.abs() * 0.01 + 0.001)
                line[c] = f"{ok.mean():.1%}"
            print(line)


if __name__ == "__main__":
    main()
