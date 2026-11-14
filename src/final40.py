"""final40 — 확률 예측 평가와 다중비교 보정 (심사 대비 보강, 탐색적).

입력: results/final27_preds.csv (본문 <표 5>·<표 6>의 1월 예측, 2019–2024 채점 행)
1) 보정(calibration): LightGBM은 class_weight="balanced"로 학습해 확률 수준이 부풀어 있다.
   예측 연도 Y의 확률은 Y 이전 채점 연도의 표본 밖 예측으로 맞춘 로지스틱 재보정(Platt)으로 바꾼다.
   이전 연도가 없는 2019년은 빼고 2020–2024만 확률 지표를 낸다(미래 정보 없이 재보정).
   지표: Brier, 로그 점수, Brier skill(기준: 학습 구간 사건률), 보정 기울기·절편, ECE(10분위).
2) 상위 k 적중 수의 수준 신뢰구간과 조건 간 Brier·로그 점수 차이 신뢰구간(기업 단위 부트스트랩 1,000회).
3) 다중비교: 주가설은 연구질문 1의 1월 PR-AUC 누출(A − B). 나머지 PR-AUC 비교 6개는 Holm 보정.
출력: results/final40_prob.csv, final40_calib.csv, final40_diff.csv, final40_topk.csv, final40_holm.csv, final40.log
실행: 저장소 루트에서 `python src/final40.py`
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

P = pd.read_csv("results/final27_preds.csv", dtype={"corp_code": str})
CONDS = ["A", "B", "B+", "A+", "B+시장", "A+시장"]
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def recalibrate(df):
    """연도 Y의 확률을 Y 이전 채점 연도로 맞춘 Platt 재보정으로 바꾼다."""
    out = {}
    for c in CONDS:
        q = np.full(len(df), np.nan)
        for Y in sorted(df.pred_year.unique()):
            past, now = (df.pred_year < Y).values, (df.pred_year == Y).values
            if past.sum() == 0:
                continue
            lr = LogisticRegression(C=1e6, max_iter=1000)
            lr.fit(logit(df.loc[past, c].values).reshape(-1, 1), df.loc[past, "y"].values)
            q[now] = lr.predict_proba(logit(df.loc[now, c].values).reshape(-1, 1))[:, 1]
        out[c] = q
    return pd.DataFrame(out, index=df.index)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def logscore(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calib_slope(y, p):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def ece(y, p, bins=10):
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    d = pd.DataFrame({"y": y, "p": p, "b": q})
    g = d.groupby("b").agg(n=("y", "size"), obs=("y", "mean"), pred=("p", "mean"))
    return float((g.n * (g.obs - g.pred).abs()).sum() / g.n.sum()), g


def main():
    rng = np.random.default_rng(42)
    log = []
    Q = recalibrate(P)
    ok = Q["A"].notna().values
    D, R = P[ok].reset_index(drop=True), Q[ok].reset_index(drop=True)
    y = D.y.values
    base = np.array([P.loc[P.pred_year < yr, "y"].mean() for yr in D.pred_year])  # 이전 연도 사건률

    rows, calib = [], []
    for c in CONDS:
        p = R[c].values
        e, g = ece(y, p)
        s, i = calib_slope(y, p)
        rows.append({"조건": c, "행": len(y), "사건": int(y.sum()),
                     "PR-AUC": round(average_precision_score(y, p), 4),
                     "Brier": round(brier(y, p), 5), "로그점수": round(logscore(y, p), 4),
                     "Brier skill": round(1 - brier(y, p) / brier(y, base), 4),
                     "보정기울기": round(s, 3), "보정절편": round(i, 3), "ECE": round(e, 4)})
        g = g.reset_index().assign(조건=c)
        calib.append(g)
    prob = pd.DataFrame(rows)
    calib = pd.concat(calib)

    # 부트스트랩: 기업 단위
    idx_by = pd.Series(range(len(D))).groupby(D.corp_code.values).indices
    keys = list(idx_by)
    pairs = [("A", "B"), ("B+", "B"), ("B+", "A"), ("A+", "B+"), ("B+시장", "B+"), ("B+시장", "A"), ("A+시장", "B+시장")]
    draws = {pr: {"Brier": [], "로그점수": []} for pr in pairs}
    for _ in range(1000):
        pick = rng.integers(0, len(keys), len(keys))
        ix = np.concatenate([idx_by[keys[j]] for j in pick])
        yy = y[ix]
        if yy.sum() < 5:
            continue
        for a, b in pairs:
            pa, pb = R[a].values[ix], R[b].values[ix]
            draws[(a, b)]["Brier"].append(brier(yy, pa) - brier(yy, pb))
            draws[(a, b)]["로그점수"].append(logscore(yy, pa) - logscore(yy, pb))
    diff = []
    for a, b in pairs:
        for k, f in (("Brier", brier), ("로그점수", logscore)):
            arr = np.array(draws[(a, b)][k])
            obs = f(y, R[a].values) - f(y, R[b].values)
            lo, hi = np.percentile(arr, [2.5, 97.5])
            diff.append({"비교": f"{a} − {b}", "지표": k, "차이(낮을수록 좋음)": round(obs, 5),
                         "하한": round(lo, 5), "상한": round(hi, 5), "a가 낫다(<0) 비율": round(float((arr < 0).mean()), 3)})
    diff = pd.DataFrame(diff)

    # 상위 k 적중 수준의 신뢰구간 (원 확률 순위, 2019–2024 전체)
    yall = P.y.values
    idx_all = pd.Series(range(len(P))).groupby(P.corp_code.values).indices
    kall = list(idx_all)
    topk = []
    for c in CONDS:
        pc = P[c].values
        obs = {k: int(yall[np.argsort(-pc)[:k]].sum()) for k in (50, 100)}
        dr = {50: [], 100: []}
        for _ in range(1000):
            pick = rng.integers(0, len(kall), len(kall))
            ix = np.concatenate([idx_all[kall[j]] for j in pick])
            o = np.argsort(-pc[ix])
            for k in (50, 100):
                dr[k].append(int(yall[ix][o[:k]].sum()))
        for k in (50, 100):
            lo, hi = np.percentile(dr[k], [2.5, 97.5])
            topk.append({"조건": c, "k": k, "적중": obs[k], "하한": lo, "상한": hi})
    topk = pd.DataFrame(topk)

    # Holm: final27_ci의 PR-AUC 비교. 주가설(A − B)은 따로 둔다.
    ci = pd.read_csv("results/final27_ci.csv")
    ci = ci[ci.지표 == "PR-AUC"].copy()
    f_ = ci["0초과"]
    ci["p(양측)"] = np.maximum(2 * np.minimum(f_, 1 - f_), 1 / 1000)
    ci["역할"] = np.where(ci.비교 == "A − B", "주가설", "부가설")
    sec = ci[ci.역할 == "부가설"].sort_values("p(양측)")
    m = len(sec)
    adj, run = [], 0.0
    for r, p in enumerate(sec["p(양측)"].values):
        run = max(run, min(1.0, (m - r) * p))
        adj.append(run)
    sec["Holm 보정 p"] = adj
    holm = pd.concat([ci[ci.역할 == "주가설"].assign(**{"Holm 보정 p": ci.loc[ci.비교 == "A − B", "p(양측)"].values}), sec])
    holm["0.05에서 유의"] = holm["Holm 보정 p"] < 0.05
    holm = holm[["비교", "의미", "역할", "격차", "하한", "상한", "p(양측)", "Holm 보정 p", "0.05에서 유의"]]

    prob.to_csv("results/final40_prob.csv", index=False, encoding="utf-8-sig")
    calib.to_csv("results/final40_calib.csv", index=False, encoding="utf-8-sig")
    diff.to_csv("results/final40_diff.csv", index=False, encoding="utf-8-sig")
    topk.to_csv("results/final40_topk.csv", index=False, encoding="utf-8-sig")
    holm.to_csv("results/final40_holm.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    txt = "\n\n".join(["== 확률 지표 (재보정, 2020–2024)", prob.to_string(index=False),
                       "== 조건 간 차이", diff.to_string(index=False),
                       "== 상위 k 적중 (2019–2024)", topk.to_string(index=False),
                       "== Holm", holm.to_string(index=False)])
    open("results/final40.log", "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
