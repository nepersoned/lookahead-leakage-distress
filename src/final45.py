"""final45 — 4월 예측의 이상치: 이미 공개된 FY(Y−1)을 버린 4개월 시차(L4)가 왜 B보다 높은가 (탐색적).

4월 1일 모집단은 1–3월에 지정된 기업(대부분 FY(Y−1) 결산 결과로 지정)을 이미 뺀 집단이다.
가설 1  그 결과 FY(Y−1)의 나쁜 수치는 '지정되지 않고 남은' 기업에서 위험 신호가 약해지고,
        더 오래된 FY(Y−2)가 남은 위험(추세)을 더 잘 담는다.
가설 2  두 연도의 정보가 서로 다르다면, 함께 넣은 조건(B + FY(Y−2))이 둘 다보다 높아야 한다.
조건 (4월 예측, 같은 행)
    B        T 이전 실제 접수된 최신 사업보고서 (약 90%가 FY(Y−1))
    L4       FY(Y−2) (결산 후 4개월 시차 규칙)
    B+L4     둘을 함께 (L4 쪽은 _prev 접미사)
    A        FY(Y−1) 전원 (미접수 10% 포함)
그리고 4월 사건(4–12월 지정)의 사유군 구성을 1월 사건과 비교한다.
출력: results/final45_score.csv, final45_ci.csv, final45_groups.csv, final45.log
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
START, TEST = 2017, list(range(2019, 2025))


def main():
    rng = np.random.default_rng(42)
    _, pub, labels, mc, c2s = f7.load()
    r = pd.read_csv(ea.FULL / "classic_orig.csv", dtype={"corp_code": str})
    r["corp_code"] = r.corp_code.str.zfill(8)
    A, B, M = f7.design(r, pub, labels, mc, c2s, list(range(2017, 2025)), 4)
    A, B, M = A.reset_index(drop=True), B.reset_index(drop=True), M.reset_index(drop=True)
    ri = r.set_index(["corp_code", "bsns_year"])
    fy2 = M.pred_year - 2
    L4 = pd.DataFrame([ri.loc[(c, f), BASE].to_dict() if (c, f) in ri.index else {k: np.nan for k in BASE}
                       for c, f in zip(M.corp_code, fy2)])
    L4 = f7.refresh_mcap(L4, M, r, mc, c2s, fy2, 4)
    X = {"A": A, "B": B, "L4": L4, "B+L4": pd.concat([B, L4.add_suffix("_prev")], axis=1)}
    P = {k: f7.rolling(v, M, START, TEST) for k, v in X.items()}
    te = M.pred_year.isin(TEST).values
    y = M.loc[te, "y"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)
    score = pd.DataFrame([{"조건": k, **{a: round(float(b), 4) for a, b in dm.score(y, p).items()}} for k, p in P.items()])
    cis = []
    for a, b in [("L4", "B"), ("B+L4", "B"), ("B+L4", "L4")]:
        for rr in ea.boot_ci(y, P[a], P[b], grp, 1000, rng):
            f_ = rr["0초과"]
            cis.append({"비교": f"{a} − {b}", **rr, "p(양측)": round(max(2 * min(f_, 1 - f_), 0.002), 3)})
    cis = pd.DataFrame(cis)

    # 단변량: 4월 모집단에서 FY(Y−1)과 FY(Y−2) 주요 비율의 AUC
    from sklearn.metrics import roc_auc_score
    uni = []
    Mt = M[te].reset_index(drop=True)
    for c in ["O_NITA", "K_RETA", "O_TLTA", "O_SCORE", "KSCORE"]:
        for name, X_ in (("B(최신 접수)", B), ("FY(Y−2)", L4)):
            v = X_.loc[te, c].reset_index(drop=True)
            ok = v.notna().values
            s = roc_auc_score(Mt.y.values[ok], v.values[ok])
            uni.append({"변수": c, "연도": name, "AUC(방향 무관)": round(max(s, 1 - s), 4)})
    uni = pd.DataFrame(uni)

    # 사유군: 1월 사건 vs 4월 사건
    e1 = pd.read_csv("results/final27_events.csv", dtype={"corp_code": str})
    grp1 = e1.사유군.value_counts(normalize=True).round(3).rename("1월 사건 비율")
    e4 = e1[e1.month >= 4]
    grp4 = e4.사유군.value_counts(normalize=True).round(3).rename("4–12월 지정 비율")
    groups = pd.concat([grp1, grp4], axis=1)

    score.to_csv("results/final45_score.csv", index=False, encoding="utf-8-sig")
    cis.to_csv("results/final45_ci.csv", index=False, encoding="utf-8-sig")
    groups.to_csv("results/final45_groups.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    txt = "\n\n".join([score.to_string(index=False), cis[cis.지표 == "PR-AUC"].to_string(index=False),
                       uni.to_string(index=False), groups.to_string()])
    open("results/final45.log", "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
