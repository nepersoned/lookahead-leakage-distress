"""라벨을 상장폐지로 바꿔도 누출이 나타나는가 — 관리종목 지정에만 해당하는 현상이 아님을 보인다.

라벨: 비자발적 상장폐지 (기업별 최초 날짜)
    KIND 관리종목 공시 중 '상장폐지' — 피흡수합병·유가증권시장 이전·스팩·신주인수권·기간만료·완전자회사 편입 제외
    DART 거래소 공시 '주권매매거래정지해제(상장폐지에따른정리매매개시)'·'상장폐지결정'(이전상장 제외)
    두 출처의 합집합에서 기업별 가장 이른 날짜를 쓴다.
모집단: 상장폐지 기업 대부분이 관리종목을 거치므로 관리종목 상태로 제외하지 않는다.
설계·모형·채점은 본문(final27 A·B)과 같다. 예측 시점 1·4·7월.

출력: results/final32_labels.csv, final32_score.csv, final32_ci.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import final7 as f7
import disclosure_mock as dm
import expand_analysis as ea
import gate3e_full as g
from dart import Dart

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"
OUT = ROOT / "results"
START, TEST = 2017, list(range(2019, 2025))
MONTHS = [1, 4, 7]
DROP = r"신주인수권|피흡수합병|유가증권시장|스팩|SPAC|기간만료|완전자회사"


def delist_labels() -> pd.DataFrame:
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes.stock_code.astype(str).str.strip().str.zfill(6)
    s2c = (codes[codes.stock_code.str.len() == 6].drop_duplicates("stock_code")
           .set_index("stock_code")["corp_code"])

    k = pd.concat([pd.read_csv(FULL / f, dtype=str) for f in ("kind_events_kosdaq_old.csv", "kind_events.csv")],
                  ignore_index=True).drop_duplicates("acpt_no")
    k = k[k.event == "상장폐지"]
    nm = k.report_nm.str.replace(r"\s", "", regex=True)
    k = k[~(nm.str.contains(DROP) | k.reason.isin(["기간만료", "합병소멸", "스팩요건"])
            | k.corp_name.str.contains("스팩|기업인수목적", na=False))]
    import kind_codes
    k = kind_codes.fill(k)                # 외국 기업의 빈 종목코드를 회사명으로 채운다
    k = pd.DataFrame({"corp_code": k.stock_code.map(s2c),
                      "event_dt": pd.to_datetime(k.rcept_dt.str[:10], errors="coerce"), "src": "KIND"})

    f = pd.read_csv(FULL / "filings_all.csv", dtype=str, usecols=["corp_code", "rcept_dt", "report_nm"])
    nm = f.report_nm.str.replace(r"\s", "", regex=True)
    f = f[nm.str.contains("정리매매개시") | (nm.str.startswith("상장폐지결정") & ~nm.str.contains("이전상장"))]
    f = pd.DataFrame({"corp_code": f.corp_code.str.zfill(8),
                      "event_dt": pd.to_datetime(f.rcept_dt.str[:8], format="%Y%m%d", errors="coerce"),
                      "src": "DART"})

    lab = pd.concat([k, f]).dropna()
    lab["corp_code"] = lab.corp_code.astype(str).str.zfill(8)
    lab = lab.sort_values("event_dt").groupby("corp_code", as_index=False).first()
    return lab


def main():
    rng = np.random.default_rng(42)
    _, pub, _, mc, c2s = f7.load()
    ratios = pd.read_csv(FULL / "classic_orig.csv", dtype={"corp_code": str})
    ratios["corp_code"] = ratios.corp_code.str.zfill(8)
    labels = delist_labels()
    OUT.mkdir(exist_ok=True)
    labels.to_csv(OUT / "final32_labels.csv", index=False, encoding="utf-8-sig")
    print(f"상장폐지 라벨 {len(labels)}개사", labels.event_dt.dt.year.value_counts().sort_index().to_dict(), flush=True)

    g.in_status = lambda code, T: False          # 관리종목 상태로 모집단을 거르지 않는다
    scores, cis = [], []
    for month in MONTHS:
        A, B, M = f7.design(ratios, pub, labels[["corp_code", "event_dt"]], mc, c2s,
                            list(range(2017, 2025)), month)
        te = M.pred_year.isin(TEST).values
        y = M.loc[te, "y"].values
        grp = M.loc[te, "corp_code"].reset_index(drop=True)
        P = {"A": f7.rolling(A, M, START, TEST), "B": f7.rolling(B, M, START, TEST)}
        for k, p in P.items():
            scores.append({"예측월": month, "조건": k, "채점행": int(te.sum()), "사건": int(y.sum()),
                           **{a: round(float(b), 4) for a, b in dm.score(y, p).items()}})
            print(scores[-1], flush=True)
        for r in ea.boot_ci(y, P["A"], P["B"], grp, 1000, rng):
            f_ = r["0초과"]
            cis.append({"예측월": month, "비교": "A−B", **r, "p(양측)": round(max(2 * min(f_, 1 - f_), 0.002), 3)})
    pd.DataFrame(scores).to_csv(OUT / "final32_score.csv", index=False, encoding="utf-8-sig")
    ci = pd.DataFrame(cis)
    ci.to_csv(OUT / "final32_ci.csv", index=False, encoding="utf-8-sig")
    print(ci[ci.지표 == "PR-AUC"].to_string(), flush=True)


if __name__ == "__main__":
    main()
