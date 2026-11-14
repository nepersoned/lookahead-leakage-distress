"""final41 — 실시간 자료(vintage) 분해: 관행 누출을 '공시 시차'와 '정정' 두 성분으로 나눈다 (탐색적).

재무 수치 두 벌
    최초본  data/full/classic_orig.csv  사업보고서 최초 공시본 원문 (본문 설계)
    정정본  data/full/classic_full.csv  DART API 값 (나중 정정이 반영된 현재값)
조건 (1월 1일 예측, 같은 행·같은 모형)
    A정정  결산일 정렬 + 정정본   ← 흔한 관행 (API로 받은 FY(Y−1))
    A      결산일 정렬 + 최초본   ← 본문 A
    B      접수일 정렬 + 최초본   ← 본문 B (예측 시점에 실제로 볼 수 있던 값)
    B정정  접수일 정렬 + 정정본   ← 시점은 지켰지만 나중 정정값을 쓴 경우
분해: A정정 − B = (A정정 − A) 정정 성분 + (A − B) 시차 성분.  B정정 − B = 시점을 지켜도 남는 정정 누출.
행: 두 벌 모두 A·B가 있는 기업·연도만(교집합). 기업 단위 부트스트랩 1,000회.
출력: results/final41_score.csv, final41_ci.csv, final41_diffrate.csv, final41.log
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import final7 as f7
import disclosure_mock as dm
import expand_analysis as ea
from final4 import BASE

warnings.filterwarnings("ignore")
FULL = ea.FULL
START, TEST = 2017, list(range(2019, 2025))
KEY = ["corp_code", "pred_year"]


def main():
    rng = np.random.default_rng(42)
    api, pub, labels, mc, c2s = f7.load()  # API 정정본 (O_SCORE 재계산 포함)
    api["corp_code"] = api.corp_code.astype(str).str.zfill(8)
    orig = pd.read_csv(FULL / "classic_orig.csv", dtype={"corp_code": str})
    orig["corp_code"] = orig.corp_code.str.zfill(8)

    Ao, Bo, Mo = f7.design(orig, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    Aa, Ba, Ma = f7.design(api, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    ko = Mo[KEY].astype(str).agg("|".join, axis=1)
    ka = Ma[KEY].astype(str).agg("|".join, axis=1)
    common = sorted(set(ko) & set(ka))
    io = pd.Series(range(len(ko)), index=ko.values).loc[common].values
    ia = pd.Series(range(len(ka)), index=ka.values).loc[common].values
    M = Mo.iloc[io].reset_index(drop=True)
    # 같은 행에서 lag(B의 연도)가 두 벌에서 같아야 비교가 맞다
    same_lag = (Mo.lag.values[io] == Ma.lag.values[ia])
    X = {"A정정": Aa.iloc[ia][BASE].reset_index(drop=True), "A": Ao.iloc[io][BASE].reset_index(drop=True),
         "B": Bo.iloc[io][BASE].reset_index(drop=True), "B정정": Ba.iloc[ia][BASE].reset_index(drop=True)}
    keep = same_lag
    M = M[keep].reset_index(drop=True)
    X = {k: v[keep].reset_index(drop=True) for k, v in X.items()}
    # 결측 차이와 값 정정을 가르기: 최초본의 빈칸만 정정본으로 채운 A (값이 다른 칸은 최초본 유지)
    X["A보완"] = X["A"].fillna(X["A정정"])
    X["B보완"] = X["B"].fillna(X["B정정"])

    # 정정으로 값이 달라진 비율 (A 쪽, 주요 비율)
    d = (X["A정정"] - X["A"]).abs()
    ev = M.y.values == 1
    anyd = (d > 1e-6).any(axis=1).values
    print(f"값이 하나라도 정정된 행 비율 — 지정 {anyd[ev].mean():.3f} / 정상 {anyd[~ev].mean():.3f} "
          f"(지정 {int(ev.sum())}행)", flush=True)
    rate = pd.DataFrame({"변수": BASE,
                         "지정 기업 달라진 비율": [(d[c][ev] > 1e-6).mean().round(3) for c in BASE],
                         "정상 기업 달라진 비율": [(d[c][~ev] > 1e-6).mean().round(3) for c in BASE],
                         "달라진 행 비율(|차이|>1e-6)": [(d[c] > 1e-6).mean().round(3) for c in BASE],
                         "A 결측": [X["A"][c].isna().mean().round(3) for c in BASE],
                         "A정정 결측": [X["A정정"][c].isna().mean().round(3) for c in BASE]})

    P = {k: f7.rolling(v, M, START, TEST) for k, v in X.items()}
    te = M.pred_year.isin(TEST).values
    y = M.loc[te, "y"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)
    score = [{"조건": k, "채점": int(te.sum()), "사건": int(y.sum()),
              **{a: round(float(b), 4) for a, b in dm.score(y, p).items()}} for k, p in P.items()]
    cis = []
    for a, b, what in [("A정정", "B", "관행 전체 누출"), ("A", "B", "시차 성분"),
                       ("A정정", "A", "정정 성분(결산일 정렬)"), ("B정정", "B", "정정 성분(접수일 정렬)"),
                       ("A보완", "A", "그중 결측 채움 몫"), ("A정정", "A보완", "그중 값 정정 몫"),
                       ("A보완", "B보완", "시차 성분(결측 채운 뒤)")]:
        for r in ea.boot_ci(y, P[a], P[b], grp, 1000, rng):
            f_ = r["0초과"]
            cis.append({"비교": f"{a} − {b}", "의미": what, **r,
                        "p(양측)": round(max(2 * min(f_, 1 - f_), 0.002), 3)})
    score, cis = pd.DataFrame(score), pd.DataFrame(cis)
    score.to_csv("results/final41_score.csv", index=False, encoding="utf-8-sig")
    cis.to_csv("results/final41_ci.csv", index=False, encoding="utf-8-sig")
    rate.to_csv("results/final41_diffrate.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    txt = "\n\n".join([f"교집합 행 {len(common):,} / lag 일치 {int(keep.sum()):,}",
                       score.to_string(index=False), cis[cis.지표 == "PR-AUC"].to_string(index=False),
                       rate.to_string(index=False)])
    open("results/final41.log", "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
