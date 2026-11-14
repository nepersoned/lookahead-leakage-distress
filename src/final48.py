"""final48 — 사건 유형별 누출을 3분기보고서를 준 B′ 기준으로 다시 잰다 (final46 + final47, 탐색적).

과제·제외 규칙은 final46과 같다(다른 사유로 지정된 기업·연도는 뺀다). 조건은 A, B, B′(B + 3분기 변수).
출력: results/final48_tasks.csv, final48.log
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import final7 as f7
import final27 as f27
import final47 as f47
import disclosure_mock as dm
import expand_analysis as ea

warnings.filterwarnings("ignore")
START, TEST = 2017, list(range(2019, 2025))


def main():
    rng = np.random.default_rng(42)
    _, pub, labels, mc, c2s = f7.load()
    r = pd.read_csv(ea.FULL / "classic_orig.csv", dtype={"corp_code": str})
    r["corp_code"] = r.corp_code.str.zfill(8)
    A, B, M = f7.design(r, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    A, B, M = A.reset_index(drop=True), B.reset_index(drop=True), M.reset_index(drop=True)
    Q, _ = f47.q3_features(M, B)
    Bq = pd.concat([B, Q], axis=1)
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
    rows = []
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
        P = {k: f7.rolling(X[keep].reset_index(drop=True), Mk, START, TEST) for k, X in (("A", A), ("B", B), ("B′", Bq))}
        s = {k: dm.score(yt, p)["PR-AUC"] for k, p in P.items()}
        for a, b in (("A", "B"), ("A", "B′")):
            c = {x["지표"]: x for x in ea.boot_ci(yt, P[a], P[b], g, 1000, rng)}["PR-AUC"]
            rows.append({"과제": name, "사건": int(yt.sum()), "비교": f"{a} − {b}",
                         "A": round(s["A"], 4), "B": round(s["B"], 4), "B′": round(s["B′"], 4),
                         "누출": c["격차"], "하한": c["하한"], "상한": c["상한"],
                         "상대": round((s[a] - s[b]) / s[b], 3)})
        print(rows[-2:], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("results/final48_tasks.csv", index=False, encoding="utf-8-sig")
    open("results/final48.log", "w", encoding="utf-8").write(t.to_string(index=False) + "\n")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
