"""최초 제출본 사업보고서 원문에서 재무 수치를 뽑는다.

문제: DART 재무제표 API(fnlttSinglAcntAll)는 기업코드·사업연도만 받고 **최신 정정본**
수치를 돌려준다. 예측일 T 이전에 최초 공시된 사실만 확인했어도, 수치에는 T 이후의 정정이
섞인다(채점 행의 4.6%, 사건 기업의 14.7%).

해결: 원문(document.xml)은 그 접수번호 시점의 문서다. 「III. 재무에 관한 사항」의
`1. 요약재무정보` 표에 당기·전기·전전기 3개 연도가 함께 실린다. 여기서 직접 읽는다.

원문 ZIP은 텍스트 작업(t1_parse.py) 때 받아 둔 data/cache/docs/ 캐시를 그대로 쓴다.
API 재호출 없음.

사용: python src/orig_fin.py [--limit 50] [--check]
    --check  정정이 없는 기업·연도에서 API 값과 대조해 파서 정확도를 낸다
출력: data/full/orig_fin.csv
"""
from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "cache" / "docs"
FULL = ROOT / "data" / "full"

# 행 이름 → 표준 항목. 위에서부터 먼저 걸리는 것으로 본다
FIELDS = [
    ("cur_assets", r"^\[?유동자산"),
    ("noncur_assets", r"^\[?비유동자산"),
    ("assets", r"^자산총계"),
    ("cur_liab", r"^\[?유동부채"),
    ("noncur_liab", r"^\[?비유동부채"),
    ("liabilities", r"^부채총계"),
    ("retained", r"이익잉여금|결손금"),
    ("equity", r"^자본총계"),
    ("revenue", r"^(매출액|영업수익|수익\(매출액\))"),
    ("op_income", r"^영업이익"),
    ("pretax", r"법인세.*차감전"),
    ("net_income", r"^(당기순이익|연결총당기순이익|당기순손익)"),
]
NUM = re.compile(r"^\(?-?[\d,]+\)?$")
YEAR = re.compile(r"(20\d{2})\s*(?:년|연도|\.|/|-)")
UNIT = re.compile(r"단위\s*:?\s*(원|천원|백만원|십억원|억원)")
SCALE = {"원": 1, "천원": 1e3, "백만원": 1e6, "억원": 1e8, "십억원": 1e9}


def _num(s: str):
    s = s.replace(" ", "")
    if not NUM.match(s):
        return None
    neg = s.startswith("(") or s.startswith("-")
    v = float(re.sub(r"[^\d]", "", s) or 0)
    return -v if neg else v


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html)).strip()


def read_xml(path: Path) -> str | None:
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        return None
    name = max(z.namelist(), key=lambda n: z.getinfo(n).file_size)
    blob = z.read(name)
    for enc in ("utf-8", "cp949"):     # 2021년 이전은 선언과 달리 EUC-KR이다
        try:
            return blob.decode(enc)
        except UnicodeDecodeError:
            continue
    return blob.decode("cp949", "replace")


def parse(xml: str) -> dict[int, dict]:
    """요약재무정보 표 → {회계연도: {항목: 값}}. 표 앞의 단위 표기로 원 단위로 맞춘다."""
    for m in re.finditer("요약재무정보", xml):
        seg = xml[m.start():m.start() + 150_000]
        scale = 1.0
        for tab in re.findall(r"<TABLE.*?</TABLE>", seg, re.S)[:12]:
            flat = _text(tab)
            u = UNIT.search(flat)
            if u:
                scale = SCALE[u.group(1)]
            rows = re.findall(r"<TR[^>]*>.*?</TR>", tab, re.S)
            if len(rows) < 5:
                continue
            cells = [[_text(c) for c in re.findall(r"<T[DEH][^>]*>.*?</T[DEH]>", r, re.S)]
                     for r in rows]
            years: list[int] = []
            for row in cells[:3]:
                found = [int(YEAR.search(c).group(1)) for c in row if YEAR.search(c)]
                if len(found) >= 2:
                    years = found
                    break
            if not years:
                continue
            vals: dict[int, dict] = {y: {} for y in years}
            for row in cells:
                if len(row) < 2:
                    continue
                label = row[0].replace(" ", "").lstrip(".ㆍ·▪-")
                nums = [v for v in (_num(c) for c in row[1:]) if v is not None]
                if len(nums) < len(years):
                    continue
                for key, pat in FIELDS:
                    if re.search(pat, label):
                        for y, v in zip(years, nums[:len(years)]):
                            vals[y].setdefault(key, v * scale)
                        break
            if any(len(v) >= 5 for v in vals.values()):
                return {y: v for y, v in vals.items() if len(v) >= 5}
    return {}


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
        xml = read_xml(DOCS / f"{rc}.zip")
        got = parse(xml) if xml else {}
        if not got:
            fail += 1
        for y, v in got.items():
            rows.append({"corp_code": code, "bsns_year": y, "src_rcept": rc,
                         "src_fy": int(fy), **v})
        if n % 500 == 0:
            print(f"  {n}/{len(idx)} 실패 {fail}", flush=True)

    df = pd.DataFrame(rows)
    # 같은 (기업, 연도)가 여러 보고서에 나오면 가장 이른 보고서(=그 해 최초 공시)를 쓴다
    df = df.sort_values("src_rcept").drop_duplicates(["corp_code", "bsns_year"])
    print(f"\n문서 {len(idx):,}건 중 파싱 실패 {fail}건 ({fail/len(idx):.1%})")
    print(f"추출 {len(df):,}행 / {df.corp_code.nunique():,}개사 "
          f"{int(df.bsns_year.min())}~{int(df.bsns_year.max())}")
    print((df.drop(columns=["corp_code", "src_rcept"]).isna().mean() * 100).round(1).to_string())
    if not a.limit:
        df.to_csv(FULL / "orig_fin.csv", index=False, encoding="utf-8-sig")

    if a.check:
        api = pd.read_csv(FULL / "ratios_full.csv", dtype={"corp_code": str})
        api["corp_code"] = api.corp_code.str.zfill(8)
        rep = pd.read_csv(FULL / "reports_full.csv", dtype={"corp_code": str})
        rep["corp_code"] = rep.corp_code.str.zfill(8)
        amd = rep[(rep.report_type == "사업") &
                  rep.is_amended.astype(str).str.lower().isin(["true", "1"])]
        amended = set(zip(amd.corp_code, amd.bsns_year))
        # ratios_full은 비율만 있으므로 같은 비율로 만들어 비교한다
        d2 = df.assign(roa=df.net_income / df.assets,
                       asset_turnover=df.revenue / df.assets,
                       retained_to_ta=df.retained / df.assets,
                       debt_ratio=df.liabilities / df.equity)
        cols = ["roa", "asset_turnover", "retained_to_ta", "debt_ratio"]
        m = d2.merge(api[["corp_code", "bsns_year"] + cols], on=["corp_code", "bsns_year"],
                     suffixes=("_doc", "_api"))
        m["정정있음"] = [tuple(k) in amended for k in zip(m.corp_code, m.bsns_year)]
        print("\n=== 원문 파싱값 vs API값 ===")
        for grp, d in m.groupby("정정있음"):
            line = {"정정 있음" if grp else "정정 없음": len(d)}
            for c in cols:
                a_, b_ = d[f"{c}_doc"], d[f"{c}_api"]
                ok = (a_ - b_).abs() <= (b_.abs() * 0.01 + 0.001)
                line[c] = f"{ok.mean():.1%}"
            print(line)


if __name__ == "__main__":
    main()
