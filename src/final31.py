"""고정 시차 규칙 대 실제 접수일 — "결산 후 N개월이면 공개됐다고 보는" 해외 관행으로 충분한가.

같은 행·같은 라벨에서 재무제표 연도만 바꿔 끼운다.
    A      FY(Y-1)                          결산일 정렬 (관행)
    B      T 이전에 실제 접수된 최신 사업보고서     접수일 정렬 (본문)
    L3     결산일 + 3개월(3/31) < T 이면 FY(Y-1), 아니면 FY(Y-2)
    L4     결산일 + 4개월(4/30) 기준
    L6     결산일 + 6개월(6/30) 기준  (Fama-French 1992 · Shumway 2001)

보는 것
    ① 고정 시차가 아직 접수되지 않은 재무제표를 쓰는 행의 비율과 그 행의 사건률 (누출)
    ② 고정 시차가 이미 접수된 재무제표를 버리는 행의 비율 (정보 손실)
    ③ PR-AUC: L − B (양수면 누출로 부풀려짐, 음수면 정보 손실)

출력: results/final31_rows.csv, final31_score.csv, final31_ci.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import final7 as f7
import disclosure_mock as dm
import expand_analysis as ea

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
START, TEST = 2017, list(range(2019, 2025))
MONTHS = [1, 4, 5, 7]
LAGS = {"L3": (3, 31), "L4": (4, 30), "L6": (6, 30)}     # 결산 다음 해 (월, 일)


def fixed_fy(M, month, mm, dd):
    """고정 시차 규칙이 T 시점에 '공개됐다'고 보는 최신 회계연도."""
    y = M.pred_year
    T = pd.to_datetime(dict(year=y, month=month, day=1))
    due = pd.to_datetime(dict(year=y, month=mm, day=dd))          # FY(Y-1)의 가정 공개일
    return np.where(due < T, y - 1, y - 2)


def swap(ratios, M, fy, mc, c2s, month):
    r = ratios.set_index(["corp_code", "bsns_year"])
    X = pd.DataFrame([r.loc[(c, f), f7.BASE].to_dict() if (c, f) in r.index else {}
                      for c, f in zip(M.corp_code, fy)], columns=f7.BASE).reset_index(drop=True)
    return f7.refresh_mcap(X, M, ratios, mc, c2s, pd.Series(fy), month)


def main():
    rng = np.random.default_rng(42)
    _, pub, labels, mc, c2s = f7.load()
    ratios = pd.read_csv(ROOT / "data" / "full" / "classic_orig.csv", dtype={"corp_code": str})
    ratios["corp_code"] = ratios.corp_code.str.zfill(8)          # 본문과 같은 최초 공시본 원문 재무
    rows, scores, cis = [], [], []
    for month in MONTHS:
        A, B, M = f7.design(ratios, pub, labels, mc, c2s, list(range(2017, 2025)), month)
        fy_b = (M.pred_year - 1 - M.lag).values
        te = M.pred_year.isin(TEST).values
        y = M.loc[te, "y"].values
        grp = M.loc[te, "corp_code"].reset_index(drop=True)
        P = {"A": f7.rolling(A, M, START, TEST), "B": f7.rolling(B, M, START, TEST)}
        for name, (mm, dd) in LAGS.items():
            fy = fixed_fy(M, month, mm, dd)
            leak, stale = fy > fy_b, fy < fy_b
            rows.append({"예측월": month, "규칙": name, "채점행": int(te.sum()),
                         "미접수 재무 사용 비율(%)": round(leak[te].mean() * 100, 2),
                         "그 행의 사건률(%)": round(M.y.values[te & leak].mean() * 100, 1) if (te & leak).any() else None,
                         "전체 사건률(%)": round(y.mean() * 100, 1),
                         "접수된 재무 버림 비율(%)": round(stale[te].mean() * 100, 2)})
            print(rows[-1], flush=True)
            if not (leak[te].any() or stale[te].any()):
                P[name] = P["B"]
                continue
            X = A if (fy == M.pred_year.values - 1).all() else swap(ratios, M, fy, mc, c2s, month)
            P[name] = f7.rolling(X, M, START, TEST)
        for k, p in P.items():
            scores.append({"예측월": month, "조건": k, **{a: round(float(b), 4) for a, b in dm.score(y, p).items()}})
        for k in ["A", *LAGS]:
            for r in ea.boot_ci(y, P[k], P["B"], grp, 1000, rng):
                f_ = r["0초과"]
                cis.append({"예측월": month, "비교": f"{k}−B", **r,
                            "p(양측)": round(max(2 * min(f_, 1 - f_), 0.002), 3)})
        print(pd.DataFrame(scores)[lambda d: d.예측월 == month].to_string(), flush=True)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "final31_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(scores).to_csv(OUT / "final31_score.csv", index=False, encoding="utf-8-sig")
    ci = pd.DataFrame(cis)
    ci.to_csv(OUT / "final31_ci.csv", index=False, encoding="utf-8-sig")
    print(ci[ci.지표 == "PR-AUC"].to_string(), flush=True)


if __name__ == "__main__":
    main()
