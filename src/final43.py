"""final43 — 학습 자료 흔들림에 대한 누출 크기의 민감도 (탐색적).

부트스트랩 신뢰구간은 채점 기업의 표본 변동만 반영한다. 학습 행이 조금 바뀔 때 모형이 달라지는 몫은 들어 있지 않다.
본문 설계(final27의 A, B)에서 학습 구간 행의 1%(또는 5%)를 무작위로 빼고 다시 학습·채점하기를 20번 되풀이해
1월 PR-AUC 누출(A − B)의 분포를 본다. 채점 행은 그대로 둔다.
출력: results/final43_perturb.csv, final43.log
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import final7 as f7
import feature_screen as fs
import gate3e_full as g
import disclosure_mock as dm
import expand_analysis as ea

warnings.filterwarnings("ignore")
START, TEST = 2017, list(range(2019, 2025))


def rolling_drop(X, M, drop_mask):
    p = np.full(len(M), np.nan)
    for Y in TEST:
        tr = (M.pred_year.values >= START) & (M.pred_year.values < Y) & ~drop_mask
        te = M.pred_year.values == Y
        Xtr = X[tr]
        lo, hi = Xtr.quantile(.01), Xtr.quantile(.99)
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15, min_child_samples=10,
                                 class_weight="balanced", verbose=-1, random_state=42)
        clf.fit(Xtr.clip(lo, hi, axis=1), M.y.values[tr])
        p[te] = clf.predict_proba(X[te].clip(lo, hi, axis=1))[:, 1]
    return p[M.pred_year.isin(TEST).values]


def main():
    rng = np.random.default_rng(7)
    _, pub, labels, mc, c2s = f7.load()
    r = pd.read_csv(ea.FULL / "classic_orig.csv", dtype={"corp_code": str})
    r["corp_code"] = r.corp_code.str.zfill(8)
    A, B, M = f7.design(r, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    A, B, M = A.reset_index(drop=True), B.reset_index(drop=True), M.reset_index(drop=True)
    te = M.pred_year.isin(TEST).values
    y = M.y.values[te]
    rows = []
    base = {k: dm.score(y, rolling_drop(X, M, np.zeros(len(M), bool)))["PR-AUC"] for k, X in (("A", A), ("B", B))}
    rows.append({"빼는 비율": 0.0, "회": 0, "A": base["A"], "B": base["B"], "A − B": base["A"] - base["B"]})
    for frac in (0.01, 0.05):
        for i in range(20):
            drop = rng.random(len(M)) < frac
            pa, pb = rolling_drop(A, M, drop), rolling_drop(B, M, drop)
            sa, sb = dm.score(y, pa)["PR-AUC"], dm.score(y, pb)["PR-AUC"]
            rows.append({"빼는 비율": frac, "회": i + 1, "A": sa, "B": sb, "A − B": sa - sb})
            print(frac, i + 1, round(sa, 4), round(sb, 4), round(sa - sb, 4), flush=True)
    d = pd.DataFrame(rows).round(4)
    d.to_csv("results/final43_perturb.csv", index=False, encoding="utf-8-sig")
    s = d[d["빼는 비율"] > 0].groupby("빼는 비율")[["A", "B", "A − B"]].describe(percentiles=[.05, .5, .95]).T
    txt = f"기준(빼지 않음): {rows[0]}\n\n" + s.round(4).to_string()
    open("results/final43.log", "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
