"""감사인·감사의견 수집 — 공시 행태 변수(감사인 교체)의 재료.

DART accnutAdtorNmNdAdtOpinion은 한 번 호출에 당기·전기·전전기 3년치를 준다.
그래서 2017·2020·2023년만 부르면 2015~2023년이 모두 채워진다.

시점 주의 — 전기·전전기 값은 나중 보고서에 다시 실린 것이지만, 그 해의 감사인은
그 해 사업보고서에서 이미 공개됐다. 변수를 만들 때는 반드시 **그 회계연도
사업보고서의 접수일**을 붙여 예측 시점 이전에 공개된 것만 쓴다.

출력: data/full/auditors_full.csv (corp_code, bsns_year, adtor, adt_opinion, emphs_matter)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from dart import Dart, DartError

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"
OFFSET = {"당기": 0, "전기": 1, "전전기": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", default="ratios_full.csv")
    ap.add_argument("--out", default="auditors_full.csv")
    ap.add_argument("--call-years", nargs="+", type=int, default=[2017, 2020, 2023])
    a = ap.parse_args()

    firms = sorted(pd.read_csv(FULL / a.ratios, dtype={"corp_code": str})
                   ["corp_code"].str.zfill(8).unique())
    d = Dart()
    rows = []
    for i, code in enumerate(firms, 1):
        for cy in a.call_years:
            try:
                r = d.get_json("accnutAdtorNmNdAdtOpinion.json", corp_code=code,
                               bsns_year=str(cy), reprt_code="11011")
            except DartError:
                continue
            for x in r.get("list") or []:
                m = re.search(r"\((당기|전기|전전기)\)", x.get("bsns_year", ""))
                if not m:
                    continue
                rows.append({"corp_code": code, "bsns_year": cy - OFFSET[m.group(1)],
                             "src": m.group(1), "call_year": cy,
                             "adtor": (x.get("adtor") or "").strip(),
                             "adt_opinion": (x.get("adt_opinion") or "").strip(),
                             "emphs_matter": (x.get("emphs_matter") or "").strip()})
        if i % 100 == 0:
            print(f"  {i}/{len(firms)}", flush=True)

    df = pd.DataFrame(rows)
    # 같은 회계연도가 여러 번 나오면 그 해 자기 보고서(당기) 값을 우선한다
    df["pri"] = df["src"].map(OFFSET)
    df = (df.sort_values(["corp_code", "bsns_year", "pri"])
            .drop_duplicates(["corp_code", "bsns_year"]).drop(columns="pri"))
    df.to_csv(FULL / a.out, index=False, encoding="utf-8-sig")
    print(f"\n감사인 {len(df):,}건 / {df.corp_code.nunique():,}개사 → {a.out}")
    print(df.bsns_year.value_counts().sort_index().to_string())
    print(df.adt_opinion.value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
