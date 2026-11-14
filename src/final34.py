"""누출(A − B)의 분해 — 차이가 어디서 나오는가.

입력: results/final27_preds.csv (채점 행별 A·B 예측값), results/final27_events.csv (사건별 월·사유군)

1. 사건 부분집합별 PR-AUC 격차: 정상 행 전부 + 해당 부분집합의 사건만 남겨 A·B의 PR-AUC를 잰다.
   부분집합 — 지정 월(1-3월 / 4-7월 / 8-12월), 사유군(감사·결산 확정 / 거래소 재량 / 법적 절차 / 기타).
   기업 단위 부트스트랩 1,000회.
2. 연도별 격차 (각 연도의 행만으로)
3. 순위 이동: 각 사건이 그해 A·B 순위에서 상위 100 안에 드는지, A만 잡은 사건의 특성

출력: results/final34_subsets.csv, final34_years.csv, final34_shift.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
N_BOOT = 1000


def boot_gap(d: pd.DataFrame, rng) -> tuple[float, float, float, float]:
    firms = d.corp_code.unique()
    idx = {f: np.where(d.corp_code.values == f)[0] for f in firms}
    gaps = []
    for _ in range(N_BOOT):
        pick = rng.choice(firms, len(firms), replace=True)
        rows = np.concatenate([idx[f] for f in pick])
        y = d.y.values[rows]
        if y.sum() == 0:
            continue
        gaps.append(ap(y, d.A.values[rows]) - ap(y, d.B.values[rows]))
    g = np.array(gaps)
    frac = (g > 0).mean()
    return float(np.quantile(g, .025)), float(np.quantile(g, .975)), float(frac), max(2 * min(frac, 1 - frac), 0.002)


def main():
    rng = np.random.default_rng(42)
    P = pd.read_csv(OUT / "final27_preds.csv", dtype={"corp_code": str})
    E = pd.read_csv(OUT / "final27_events.csv", dtype={"corp_code": str})
    for d in (P, E):
        d["corp_code"] = d.corp_code.str.zfill(8)
    D = P.merge(E[["corp_code", "pred_year", "month", "사유군", "lag"]], on=["corp_code", "pred_year"], how="left")
    D["월구간"] = pd.cut(D.month, [0, 3, 7, 12], labels=["1-3월", "4-7월", "8-12월"])

    rows = []
    full = (ap(D.y, D.A), ap(D.y, D.B))
    rows.append({"구분": "전체", "부분집합": "전체", "사건": int(D.y.sum()), "A": full[0], "B": full[1],
                 "격차": full[0] - full[1], **dict(zip(["하한", "상한", "0초과", "p"], boot_gap(D, rng)))})
    for col in ("월구간", "사유군"):
        for v in D.loc[D.y == 1, col].dropna().unique():
            s = D[(D.y == 0) | (D[col] == v)]
            a, b = ap(s.y, s.A), ap(s.y, s.B)
            rows.append({"구분": col, "부분집합": str(v), "사건": int(s.y.sum()), "A": a, "B": b, "격차": a - b,
                         **dict(zip(["하한", "상한", "0초과", "p"], boot_gap(s, rng)))})
    S = pd.DataFrame(rows).round(4)
    S.to_csv(OUT / "final34_subsets.csv", index=False, encoding="utf-8-sig")
    print(S.to_string(index=False), flush=True)

    yrs = []
    for y, s in D.groupby("pred_year"):
        a, b = ap(s.y, s.A), ap(s.y, s.B)
        yrs.append({"연도": y, "사건": int(s.y.sum()), "A": a, "B": b, "격차": a - b})
    Y = pd.DataFrame(yrs).round(4)
    Y.to_csv(OUT / "final34_years.csv", index=False, encoding="utf-8-sig")
    print(Y.to_string(index=False), flush=True)

    D["rA"] = D.groupby("pred_year").A.rank(ascending=False)
    D["rB"] = D.groupby("pred_year").B.rank(ascending=False)
    ev = D[D.y == 1].copy()
    ev["A만"] = (ev.rA <= 100) & (ev.rB > 100)
    ev["B만"] = (ev.rB <= 100) & (ev.rA > 100)
    sh = []
    for col in ("월구간", "사유군"):
        t = ev.groupby(col, observed=True).agg(사건=("y", "size"), A만=("A만", "sum"), B만=("B만", "sum"),
                                                A순위중앙=("rA", "median"), B순위중앙=("rB", "median")).reset_index()
        t.insert(0, "구분", col)
        t = t.rename(columns={col: "부분집합"})
        sh.append(t)
    Sh = pd.concat(sh)
    Sh.to_csv(OUT / "final34_shift.csv", index=False, encoding="utf-8-sig")
    print(Sh.to_string(index=False), flush=True)
    print(f"A만 상위100: {int(ev.A만.sum())}건, B만: {int(ev.B만.sum())}건", flush=True)


if __name__ == "__main__":
    main()
