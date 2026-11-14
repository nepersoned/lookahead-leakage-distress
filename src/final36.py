"""final36 — 누출의 내용: A가 미리 보는 한 해 동안 재무가 얼마나 변했는가.

입력: results/final27_preds.csv (채점 행), results/final27_events.csv (사건별 월·사유군·lag),
      data/full/classic_orig.csv (최초 공시본 재무비율).
방법: 행마다 A가 쓰는 FY(Y-1)과 B가 쓰는 FY(Y-1-lag)의 비율을 비교한다.
      정상 기업은 lag=1로 둔다(1월 예측에서 A가 미공개 재무제표를 쓰는 비율 100%, <표 7>).
출력: results/final36_anatomy.csv, results/final36.log
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def main():
    R = pd.read_csv(ROOT / "data/full/classic_orig.csv", dtype={"corp_code": str})
    R["corp_code"] = R.corp_code.str.zfill(8)
    R = R.set_index(["corp_code", "bsns_year"])
    P = pd.read_csv(OUT / "final27_preds.csv", dtype={"corp_code": str})
    E = pd.read_csv(OUT / "final27_events.csv", dtype={"corp_code": str})
    for d in (P, E):
        d["corp_code"] = d.corp_code.str.zfill(8)
    D = P.merge(E[["corp_code", "pred_year", "lag", "month", "사유군"]], on=["corp_code", "pred_year"], how="left")
    D["lag"] = D.lag.fillna(1).astype(int)
    fyA, fyB = D.pred_year - 1, D.pred_year - 1 - D.lag

    def get(fy, col):
        return pd.Series([R[col].get(k, np.nan) for k in zip(D.corp_code, fy)], index=D.index)

    def flip(col):
        a, b = get(fyA, col), get(fyB, col)
        f = ((b == 0) & (a == 1)).astype(float)
        f[a.isna() | b.isna()] = np.nan
        return f

    a, b = get(fyA, "O_NITA"), get(fyB, "O_NITA")
    D["ΔNITA"] = a - b
    D["흑→적"] = ((b >= 0) & (a < 0)).astype(float)
    D.loc[a.isna() | b.isna(), "흑→적"] = np.nan
    D["ΔRETA"] = get(fyA, "K_RETA") - get(fyB, "K_RETA")
    D["ΔO_SCORE"] = get(fyA, "O_SCORE") - get(fyB, "O_SCORE")
    D["INTWO 전환"] = flip("O_INTWO")
    D["OENEG 전환"] = flip("O_OENEG")
    D["집단"] = np.where(D.y == 0, "정상",
                np.where(D.month <= 3, "1–3월 지정", np.where(D.month <= 7, "4–7월 지정", "8–12월 지정")))
    cols = ["ΔNITA", "ΔRETA", "ΔO_SCORE", "흑→적", "INTWO 전환", "OENEG 전환"]
    order = ["1–3월 지정", "4–7월 지정", "8–12월 지정", "정상"]
    g1 = D.groupby("집단")[cols].mean().loc[order]
    g1.insert(0, "n", D.groupby("집단").size().loc[order])
    ev = D[D.y == 1]
    g2 = ev.groupby("사유군")[cols].mean()
    g2.insert(0, "n", ev.groupby("사유군").size())
    g1.index.name = g2.index.name = "구분"
    res = pd.concat([g1.assign(기준="지정 월"), g2.assign(기준="사유군")]).reset_index()
    res.to_csv(OUT / "final36_anatomy.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "final36.log", "w", encoding="utf-8") as f:
        pd.set_option("display.width", 200)
        f.write(res.round(3).to_string() + "\n")
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
