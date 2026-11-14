"""final49 — 메커니즘 검증: 변수 바꿔치기 (사전 계획 results/mechanism_swap_plan.md, 커밋 662bc18).

A에서 규칙 관련 변수 R만 B 값으로 바꾼 조건과, 규칙 무관 변수 N만 바꾼 조건의 누출을 잰다.
과제·제외 규칙은 final48과 같다. 판정은 계획에 고정한 P1–P3.
출력: results/final49_swap.csv, final49.log
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import final7 as f7
import final27 as f27
import disclosure_mock as dm
import expand_analysis as ea

warnings.filterwarnings("ignore")
START, TEST = 2017, list(range(2019, 2025))
R = ["O_NITA", "O_INTWO", "O_CHIN", "K_RETA", "O_TLTA", "O_OENEG", "K_lnSLTA", "KSCORE", "O_SCORE"]
N = ["K_lnTA", "O_SIZE", "K_METL", "O_WCTA", "O_CLCA", "O_FUTL_CFO"]


def main():
    rng = np.random.default_rng(42)
    _, pub, labels, mc, c2s = f7.load()
    r = pd.read_csv(ea.FULL / "classic_orig.csv", dtype={"corp_code": str})
    r["corp_code"] = r.corp_code.str.zfill(8)
    A, B, M = f7.design(r, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    A, B, M = A.reset_index(drop=True), B.reset_index(drop=True), M.reset_index(drop=True)
    assert set(R) | set(N) == set(A.columns) and not set(R) & set(N)
    AR = A.copy(); AR[R] = B[R]
    AN = A.copy(); AN[N] = B[N]
    key = list(zip(M.corp_code, M.pred_year))
    ev = f27.reason_groups(M[M.y == 1][["corp_code", "pred_year", "y"]].copy())
    rs = dict(zip(zip(ev.corp_code, ev.pred_year), ev.사유.astype(str)))
    sb = dict(zip(zip(ev.corp_code, ev.pred_year), ev["sub"].fillna("").astype(str)))
    gm = dict(zip(zip(ev.corp_code, ev.pred_year), ev.사유군))

    def fine(k):
        r_, s_ = rs.get(k, ""), sb.get(k, "")
        if s_ == "감사의견" or "감사" in r_:
            return "감사의견"
        if s_ == "보고서미제출" or "미제출" in r_:
            return "보고서 미제출"
        if s_ == "자본잠식" or any(w in r_ for w in ("손실", "자본잠식", "매출액")):
            return "재무수치 규칙"
        return gm.get(k, "")
    fg = np.array([fine(k) if M.y.values[i] == 1 else "" for i, k in enumerate(key)])
    rows, verdict = [], {}
    for name in ["전체", "재무수치 규칙", "감사의견", "거래소 재량"]:
        if name == "전체":
            y, keep = M.y.values.copy(), np.ones(len(M), bool)
        else:
            y = ((M.y.values == 1) & (fg == name)).astype(int)
            keep = ~((M.y.values == 1) & (fg != name))
        Mk = M[keep].reset_index(drop=True).assign(y=y[keep])
        te = Mk.pred_year.isin(TEST).values
        yt = Mk.y.values[te]
        g = Mk.loc[te, "corp_code"].reset_index(drop=True)
        P = {k: f7.rolling(X[keep].reset_index(drop=True), Mk, START, TEST)
             for k, X in (("A", A), ("B", B), ("A_R←B", AR), ("A_N←B", AN))}
        s = {k: dm.score(yt, p)["PR-AUC"] for k, p in P.items()}
        res = {}
        for a in ("A", "A_R←B", "A_N←B"):
            c = {x["지표"]: x for x in ea.boot_ci(yt, P[a], P["B"], g, 1000, rng)}["PR-AUC"]
            res[a] = c
            rows.append({"과제": name, "사건": int(yt.sum()), "조건": a, "PR-AUC": round(s[a], 4),
                         "B PR-AUC": round(s["B"], 4), "누출(조건−B)": c["격차"], "하한": c["하한"], "상한": c["상한"],
                         "A−B 대비 비율": round(c["격차"] / res["A"]["격차"], 3) if res["A"]["격차"] else np.nan})
        LA, LR, LN = res["A"]["격차"], res["A_R←B"]["격차"], res["A_N←B"]["격차"]
        if name == "재무수치 규칙":
            c1, c2 = LR <= 0.25 * LA, LN >= 0.75 * LA
            verdict["P1"] = {"R 바꿈 ≤ 25%": bool(c1), "N 바꿈 ≥ 75%": bool(c2),
                             "판정": "지지" if c1 and c2 else ("부분 지지" if c1 or c2 else "기각")}
        if name == "전체":
            verdict["P2"] = {"N 바꿈 누출 > R 바꿈 누출": bool(LN > LR)}
        if name in ("감사의견", "거래소 재량"):
            verdict[f"P3 {name}"] = {"모든 조건 CI가 0 포함": bool(all(res[k]["하한"] <= 0 <= res[k]["상한"] for k in res))}
        print(name, {k: round(v["격차"], 4) for k, v in res.items()}, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("results/final49_swap.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    txt = t.to_string(index=False) + "\n\n판정: " + str(verdict)
    open("results/final49.log", "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
