"""final52 — K-Score 재현 오류 수정의 영향 (탐색적).

원문(Altman, Eom & Kim 1994, NYU Salomon Center WP FIN-94-002, p.9)
    K1 = −17.862 + 1.472·LOG(TA) + 3.041·LOG(SALES/TA) + 14.839·RE/TA + 1.516·BEQ/TL   (비상장용, 장부가 자본)
    K2 = −18.696 + 1.501·LOG(TA) + 2.706·LOG(SALES/TA) + 19.760·RE/TA + 1.146·MEQ/TL   (상장용, 시가총액)
본문 코드(vars_classic.py)는 K1 계수(이인로·김동철 2015 인용, 반올림)에 시가총액/총부채(K2의 변수)를 썼다.
여기서는 상장 기업 표본이므로 K2를 원문 계수 그대로 다시 계산한다(예측 시점 직전 월말 시가총액 사용은 본문과 같음).
조건: A, B, B′ (KSCORE만 K2로 바꿈). 비교: A − B, A − B′.
출력: results/final52_score.csv, final52_ci.csv, final52.log
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


def k2(X):
    X = X.copy()
    X["KSCORE"] = -18.696 + 1.501 * X.K_lnTA + 2.706 * X.K_lnSLTA + 19.760 * X.K_RETA + 1.146 * X.K_METL
    return X


def main():
    rng = np.random.default_rng(42)
    _, pub, labels, mc, c2s = f7.load()
    r = pd.read_csv(ea.FULL / "classic_orig.csv", dtype={"corp_code": str})
    r["corp_code"] = r.corp_code.str.zfill(8)
    A, B, M = f7.design(r, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    A, B, M = k2(A.reset_index(drop=True)), k2(B.reset_index(drop=True)), M.reset_index(drop=True)
    Q, _ = f47.q3_features(M, B)
    X = {"A": A, "B": B, "B′": pd.concat([B, Q], axis=1)}
    P = {k: f7.rolling(v, M, START, TEST) for k, v in X.items()}
    te = M.pred_year.isin(TEST).values
    y = M.loc[te, "y"].values
    g = M.loc[te, "corp_code"].reset_index(drop=True)
    score = pd.DataFrame([{"조건(K2 원문 계수)": k, **{a: round(float(b), 4) for a, b in dm.score(y, p).items()}}
                          for k, p in P.items()])
    cis = []
    for a, b in (("A", "B"), ("A", "B′")):
        for rr in ea.boot_ci(y, P[a], P[b], g, 1000, rng):
            f_ = rr["0초과"]
            cis.append({"비교": f"{a} − {b}", **rr, "p(양측)": round(max(2 * min(f_, 1 - f_), 0.002), 3)})
    cis = pd.DataFrame(cis)
    score.to_csv("results/final52_score.csv", index=False, encoding="utf-8-sig")
    cis.to_csv("results/final52_ci.csv", index=False, encoding="utf-8-sig")
    txt = score.to_string(index=False) + "\n\n" + cis[cis.지표 == "PR-AUC"].to_string(index=False)
    open("results/final52.log", "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
