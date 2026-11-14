"""오류 분해 — A+는 잡고 B+는 놓친 사건이 어떤 사유인가.

가설: A+가 앞서는 이유는 손실지속·자본잠식처럼 **결산 확정 자체가 지정 사유**인 사건을
미리 본 문서로 맞히기 때문이다. 그렇다면 놓친 사건이 그 사유에 몰려 있어야 한다.
고르게 퍼져 있으면 규정 시뮬레이션 변수의 기대 효과가 작다.

조건·선별 목록·확장창은 본문(final7/final8)과 같다.
출력: results/final10_miss.csv (사건별 순위), final10_by_reason.csv (사유별 집계)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import final8 as f8
import expand_analysis as ea
import disclosure_mock as dm

warnings.filterwarnings("ignore")
ROOT = ea.ROOT


def ranks(p, yr):
    """연도 안에서의 순위(1위가 가장 위험)와 상위 백분위."""
    r = np.empty(len(p))
    pc = np.empty(len(p))
    for y in np.unique(yr):
        m = yr == y
        order = (-p[m]).argsort().argsort() + 1
        r[m] = order
        pc[m] = order / m.sum()
    return r, pc


def main():
    A, B, M, CB, CR = f8.kosdaq_design()
    sel = f8.selected()
    conds = {"A": A, "A+": pd.concat([A, CB[sel], CR], axis=1),
             "B": B, "B+": pd.concat([B, CB[sel], CR], axis=1)}
    te = M.pred_year.isin(f8.TEST).values
    Mt = M[te].reset_index(drop=True)
    P = {k: f8.f7.rolling(X, M, f8.START, f8.TEST) for k, X in conds.items()}
    for k, p in P.items():
        print(k, {a: round(float(b), 4) for a, b in dm.score(Mt.y.values, p).items()}, flush=True)

    out = Mt[["corp_code", "pred_year", "y", "lag"]].copy()
    for k, p in P.items():
        r, pc = ranks(p, Mt.pred_year.values)
        out[f"rank_{k}"] = r.astype(int)
        out[f"pct_{k}"] = pc.round(4)

    # 사건에 지정 사유 붙이기
    kind = pd.read_csv(ROOT / "data" / "full" / "kind_events.csv",
                       dtype={"stock_code": str}, parse_dates=["rcept_dt"])
    kind = kind[kind.event == "지정"].copy()
    kind["stock_code"] = kind.stock_code.str.replace(r"\.0$", "", regex=True).str.zfill(6)
    from dart import Dart
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes.stock_code.astype(str).str.strip().str.zfill(6)
    cmap = (codes[codes.stock_code.str.len() == 6].drop_duplicates("stock_code")
            .set_index("stock_code")["corp_code"])
    kind["corp_code"] = kind.stock_code.map(cmap)
    kind = kind.dropna(subset=["corp_code"])
    kind["corp_code"] = kind.corp_code.astype(str).str.zfill(8)
    kind["pred_year"] = kind.rcept_dt.dt.year
    first = (kind.sort_values("rcept_dt")
             .groupby(["corp_code", "pred_year"], as_index=False)
             .first()[["corp_code", "pred_year", "reason", "rcept_dt"]])
    ev = out[out.y == 1].merge(first, on=["corp_code", "pred_year"], how="left")
    ev["month"] = ev.rcept_dt.dt.month
    ev["reason"] = ev.reason.fillna("미상")
    ev.to_csv("results/final10_miss.csv", index=False, encoding="utf-8-sig")
    print(f"\n사건 {len(ev)}건 / 사유 미상 {int((ev.reason == '미상').sum())}건")

    ev["A+상위100"] = ev["rank_A+"] <= 100
    ev["B+상위100"] = ev["rank_B+"] <= 100
    ev["A+만"] = ev["A+상위100"] & ~ev["B+상위100"]
    ev["B+만"] = ~ev["A+상위100"] & ev["B+상위100"]
    ev["둘다놓침"] = ~ev["A+상위100"] & ~ev["B+상위100"]

    g = ev.groupby("reason").agg(
        사건=("y", "size"),
        A플러스_상위100=("A+상위100", "sum"), B플러스_상위100=("B+상위100", "sum"),
        A플러스만=("A+만", "sum"), B플러스만=("B+만", "sum"), 둘다놓침=("둘다놓침", "sum"),
        A순위중앙=("rank_A+", "median"), B순위중앙=("rank_B+", "median"),
    ).sort_values("사건", ascending=False)
    g["순위차_중앙"] = g.B순위중앙 - g.A순위중앙
    g.to_csv("results/final10_by_reason.csv", encoding="utf-8-sig")
    print("\n=== 지정 사유별 (상위 100위 안에 드는가) ===")
    print(g.to_string())

    print("\n=== 지정 월별 ===")
    mg = ev.groupby("month").agg(사건=("y", "size"), A플러스=("A+상위100", "sum"),
                                 B플러스=("B+상위100", "sum"), A플러스만=("A+만", "sum"))
    print(mg.to_string())

    fin = {"손실지속", "자본잠식", "매출액미달", "감사의견", "보고서미제표", "보고서미제출"}
    ev["결산확정형"] = ev.reason.isin({"손실지속", "자본잠식", "매출액미달"})
    print("\n=== 결산 확정이 곧 사유인 사건 vs 나머지 ===")
    print(ev.groupby("결산확정형").agg(
        사건=("y", "size"), A플러스_상위100=("A+상위100", "sum"),
        B플러스_상위100=("B+상위100", "sum"), A플러스만=("A+만", "sum"),
        A순위중앙=("rank_A+", "median"), B순위중앙=("rank_B+", "median")).to_string())


if __name__ == "__main__":
    main()
