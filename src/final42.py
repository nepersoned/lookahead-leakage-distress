"""final42 — 표본 정의의 미래 의존 점검 (탐색적).

본문 모집단은 'FY(Y−1)과 B 재무제표가 모두 있는 기업'이다. 그런데 1월 1일에는 아무도 FY(Y−1) 사업보고서를
내지 않았으므로, 그것이 '있다'는 조건은 그 기업이 이후 보고서를 냈다는 미래 정보에 기댄다.
보고서 미제출은 그 자체로 관리종목 사유이므로, 이 조건은 사건을 표본에서 빼는 방향으로 작용할 수 있다.

예측 시점 정보만으로 정한 모집단
    T 이전에 공시된 최신 사업보고서(B)가 있고 그것이 FY(Y−2) 이후일 것(상장 유지의 대리 지표),
    T 직전 관리종목 상태가 아닐 것. FY(Y−1) 존재 여부는 묻지 않는다.
비교
    본문 모집단 vs 확장 모집단의 행·사건 수, 빠졌던 행의 사건률과 사유.
    확장 모집단에서 B의 성능(정직한 모형), A의 성능(빠진 FY(Y−1)은 결측 → 결측 자체가 미래 신호).
출력: results/final42_sample.csv, final42_score.csv, final42_ci.csv, final42.log
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import final7 as f7
import disclosure_mock as dm
import expand_analysis as ea
import gate3e_full as g
from final4 import BASE

warnings.filterwarnings("ignore")
FULL = ea.FULL
START, TEST = 2017, list(range(2019, 2025))


def build_open(ratios, pub, labels, years, month=1, horizon=365):
    r = ratios.set_index(["corp_code", "bsns_year"])
    by_firm: dict[str, list] = {}
    for c, fy, d in zip(pub["corp_code"], pub["bsns_year"], pub["pub_dt"]):
        by_firm.setdefault(c, []).append((fy, d))
    ev = labels.groupby("corp_code")["event_dt"].apply(list).to_dict()
    firms = sorted(set(ratios["corp_code"]))
    A, B, M = [], [], []
    for y in years:
        T = pd.Timestamp(year=y, month=month, day=1)
        for code in firms:
            avail = [(fy, d) for fy, d in by_firm.get(code, []) if d < T]
            if not avail:
                continue
            fy_b = max(fy for fy, _ in avail)
            fy_a = y - 1
            if fy_b < y - 2 or (code, fy_b) not in r.index:
                continue
            if g.in_status(code, T):
                continue
            hit = any(T <= d < T + pd.Timedelta(days=horizon) for d in ev.get(code, []))
            has_a = (code, fy_a) in r.index
            A.append(r.loc[(code, fy_a), BASE].to_dict() if has_a else {c: np.nan for c in BASE})
            B.append(r.loc[(code, fy_b), BASE].to_dict())
            M.append({"corp_code": code, "pred_year": y, "y": int(hit), "lag": fy_a - fy_b, "has_a": int(has_a)})
    return pd.DataFrame(A), pd.DataFrame(B), pd.DataFrame(M)


def main():
    rng = np.random.default_rng(42)
    _, pub, labels, mc, c2s = f7.load()
    ratios = pd.read_csv(FULL / "classic_orig.csv", dtype={"corp_code": str})
    ratios["corp_code"] = ratios.corp_code.str.zfill(8)
    g.RATIO_COLS = BASE
    A, B, M = build_open(ratios, pub, labels, list(range(2017, 2025)))
    fy_a = M.pred_year - 1
    A = f7.refresh_mcap(A, M, ratios, mc, c2s, fy_a, 1)
    B = f7.refresh_mcap(B, M, ratios, mc, c2s, fy_a - M.lag, 1)

    te = M.pred_year.isin(TEST).values
    Mt = M[te]
    miss = Mt[Mt.has_a == 0]
    sample = {"확장 채점 행": int(te.sum()), "확장 채점 사건": int(Mt.y.sum()),
              "FY(Y−1) 없는 채점 행": len(miss), "그중 사건": int(miss.y.sum()),
              "FY(Y−1) 없는 행 사건률": round(float(miss.y.mean()), 4) if len(miss) else None,
              "FY(Y−1) 있는 행 사건률": round(float(Mt[Mt.has_a == 1].y.mean()), 4)}
    print(sample, flush=True)
    # 빠졌던 사건의 사유
    ev = pd.read_csv("results/final27_events.csv", dtype={"corp_code": str})
    lab = labels.copy()
    lab["pred_year"] = lab.event_dt.dt.year
    lab["corp_code"] = lab.corp_code.astype(str).str.zfill(8)
    mm = miss[miss.y == 1][["corp_code", "pred_year"]].merge(
        lab[["corp_code", "pred_year", "reason"]] if "reason" in lab else lab[["corp_code", "pred_year"]],
        on=["corp_code", "pred_year"], how="left")
    reasons = mm.reason.value_counts().head(10).to_dict() if "reason" in mm else {}
    print("빠졌던 사건 사유:", reasons, flush=True)

    P = {"A(결측 포함)": f7.rolling(A, M, START, TEST), "B": f7.rolling(B, M, START, TEST),
         "A+결측표시": f7.rolling(A.assign(a_missing=1 - M.has_a.values), M, START, TEST)}
    y = M.loc[te, "y"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)
    has = M.loc[te, "has_a"].values == 1
    score = []
    for k, p in P.items():
        score.append({"조건": k, "모집단": "확장", **{a: round(float(b), 4) for a, b in dm.score(y, p).items()}})
        score.append({"조건": k, "모집단": "FY(Y−1) 있는 행만", **{a: round(float(b), 4) for a, b in dm.score(y[has], p[has]).items()}})
    cis = []
    for a, b in [("A(결측 포함)", "B"), ("A+결측표시", "B")]:
        for r in ea.boot_ci(y, P[a], P[b], grp, 1000, rng):
            f_ = r["0초과"]
            cis.append({"비교": f"{a} − {b}", "모집단": "확장", **r, "p(양측)": round(max(2 * min(f_, 1 - f_), 0.002), 3)})
    score, cis = pd.DataFrame(score), pd.DataFrame(cis)
    pd.DataFrame([{**sample, "빠졌던 사건 사유": str(reasons)}]).to_csv("results/final42_sample.csv", index=False, encoding="utf-8-sig")
    score.to_csv("results/final42_score.csv", index=False, encoding="utf-8-sig")
    cis.to_csv("results/final42_ci.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    txt = "\n\n".join([str(sample), "빠졌던 사건 사유: " + str(reasons), score.to_string(index=False),
                       cis[cis.지표 == "PR-AUC"].to_string(index=False)])
    open("results/final42.log", "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
