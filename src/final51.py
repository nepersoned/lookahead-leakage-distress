"""final51 — 합성 점수(KSCORE, O_SCORE)를 빼고 구성 변수만 쓴 A·B·B′ (탐색적).

K-Score 계수(이인로·김동철 2015 인용)와 O-Score 수정판(FUTL 대체, SIZE 물가 조정 생략)의 재현 충실도가
누출 결론에 영향을 주는지 본다. 합성 점수를 빼면 계수는 결과에 들어가지 않는다.
출력: results/final51_score.csv, final51_ci.csv, final51.log
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import final7 as f7
import final47 as f47
import disclosure_mock as dm
import expand_analysis as ea

warnings.filterwarnings("ignore")
START, TEST = 2017, list(range(2019, 2025))
DROP = ["KSCORE", "O_SCORE"]


def main():
    rng = np.random.default_rng(42)
    _, pub, labels, mc, c2s = f7.load()
    r = pd.read_csv(ea.FULL / "classic_orig.csv", dtype={"corp_code": str})
    r["corp_code"] = r.corp_code.str.zfill(8)
    A, B, M = f7.design(r, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    A, B, M = A.reset_index(drop=True), B.reset_index(drop=True), M.reset_index(drop=True)
    Q, _ = f47.q3_features(M, B)
    A0, B0 = A.drop(columns=DROP), B.drop(columns=DROP)
    X = {"A": A0, "B": B0, "B′": pd.concat([B0, Q], axis=1)}
    P = {k: f7.rolling(v, M, START, TEST) for k, v in X.items()}
    te = M.pred_year.isin(TEST).values
    y = M.loc[te, "y"].values
    g = M.loc[te, "corp_code"].reset_index(drop=True)
    score = pd.DataFrame([{"조건(합성 점수 제외)": k, **{a: round(float(b), 4) for a, b in dm.score(y, p).items()}}
                          for k, p in P.items()])
    cis = []
    for a, b in (("A", "B"), ("A", "B′")):
        for rr in ea.boot_ci(y, P[a], P[b], g, 1000, rng):
            f_ = rr["0초과"]
            cis.append({"비교": f"{a} − {b}", **rr, "p(양측)": round(max(2 * min(f_, 1 - f_), 0.002), 3)})
    cis = pd.DataFrame(cis)
    score.to_csv("results/final51_score.csv", index=False, encoding="utf-8-sig")
    cis.to_csv("results/final51_ci.csv", index=False, encoding="utf-8-sig")
    txt = score.to_string(index=False) + "\n\n" + cis[cis.지표 == "PR-AUC"].to_string(index=False)
    open("results/final51.log", "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
