"""final47 — 2단계: 3분기보고서를 준 정직한 조건(B′)으로 누출을 다시 잰다.

반론: 1월 1일에는 직전 연도 3분기보고서가 이미 공개돼 있는데 B는 사업보고서(FY(Y−2))만 쓴다.
     그러면 A − B에는 'B가 쓸 수 있었는데 안 쓴 정보'의 몫이 섞인다.
조건 (1월 예측, 같은 행·같은 모형)
    B      FY(Y−2) 사업보고서 (본문)
    B′     B + 직전 연도 3분기보고서 변수 (예측 시점 이전에 접수된 것만)
    A      FY(Y−1) 사업보고서 (관행)
    A+Q    A + 같은 3분기 변수 (정보 집합을 맞춘 비교)
3분기 변수 (3분기말 재무상태표, 9개월 누적 손익)
    q_TLTA q_WCTA q_CLCA q_OENEG q_RETA q_SIZE  재무상태표 비율
    q_NITA9 q_OPTA9 q_CFOTL9                    9개월 누적 순이익·영업이익/총자산, 영업현금흐름/총부채
    q_CHIN9 q_REVG9 q_LOSS9                     전년 같은 기간 대비 순이익 변화·매출 증가율·9개월 적자 여부
    q_NITA_TTM                                  최근 12개월 순이익/총자산 = (FY(Y−2) 순이익 + 9개월 − 전년 9개월) / 3분기말 총자산
핵심 비교: A − B′(정직한 기준을 강화한 누출), A+Q − B′(같은 정보 집합), B′ − B(3분기 정보의 가치).
출력: results/final47_score.csv, final47_ci.csv, final47_cover.csv, final47.log
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import final7 as f7
import disclosure_mock as dm
import expand_analysis as ea

warnings.filterwarnings("ignore")
START, TEST = 2017, list(range(2019, 2025))


def q3_features(M, B, month=1):
    q = pd.read_csv(ea.FULL / "q3_fin.csv", dtype={"corp_code": str, "rcept_no": str})
    q["corp_code"] = q.corp_code.str.zfill(8)
    rep = pd.read_csv(ea.FULL / "reports_full.csv", dtype={"corp_code": str, "rcept_no": str})
    rep["corp_code"] = rep.corp_code.str.zfill(8)
    rep = rep[(rep.report_type == "분기") & (rep.period_end.astype(str) == "9")]
    rep["rcept_dt"] = pd.to_datetime(rep.rcept_dt)
    first = rep.groupby(["corp_code", "bsns_year"]).rcept_dt.min()
    first.index = [(c, int(y)) for c, y in first.index]
    qi = q.set_index(["corp_code", "bsns_year"])
    rows, cover = [], []
    for i, (c, y) in enumerate(zip(M.corp_code, M.pred_year)):
        T = pd.Timestamp(year=y, month=month, day=1)
        fy = y - 1
        r = {}
        dt = first.get((c, fy))
        ok = (c, fy) in qi.index and dt is not None and dt < T
        cover.append({"pred_year": y, "q3_있음": (c, fy) in qi.index, "T 이전 접수": bool(ok)})
        if ok:
            s = qi.loc[(c, fy)]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            ta, tl = s.assets, s.liabilities
            ca, cl = s.cur_assets, s.cur_liab
            dv = lambda a, b: a / b if pd.notna(a) and pd.notna(b) and b != 0 else np.nan
            r["q_TLTA"] = dv(tl, ta)
            r["q_WCTA"] = dv(ca - cl if pd.notna(ca) and pd.notna(cl) else np.nan, ta)
            r["q_CLCA"] = dv(cl, ca)
            r["q_OENEG"] = float(tl > ta) if pd.notna(tl) and pd.notna(ta) else np.nan
            r["q_RETA"] = dv(s.retained, ta)
            r["q_SIZE"] = np.log(ta) if pd.notna(ta) and ta > 0 else np.nan
            ni9, ni9p = s.net_income_9m, s.net_income_9m_prev
            r["q_NITA9"] = dv(ni9, ta)
            r["q_OPTA9"] = dv(s.op_income_9m, ta)
            r["q_CFOTL9"] = dv(s.cfo_9m, tl)
            r["q_CHIN9"] = (dv(ni9 - ni9p, abs(ni9) + abs(ni9p)) if pd.notna(ni9) and pd.notna(ni9p) else np.nan)
            r["q_REVG9"] = dv(s.revenue_9m, s.revenue_9m_prev) - 1 if pd.notna(s.revenue_9m_prev) else np.nan
            r["q_LOSS9"] = float(ni9 < 0) if pd.notna(ni9) else np.nan
            nb, sb = B.at[i, "O_NITA"], B.at[i, "O_SIZE"]
            ni_fy = nb * np.exp(sb) if pd.notna(nb) and pd.notna(sb) else np.nan
            r["q_NITA_TTM"] = dv(ni_fy + ni9 - ni9p, ta) if pd.notna(ni_fy) and pd.notna(ni9) and pd.notna(ni9p) else np.nan
        rows.append(r)
    cols = ["q_TLTA", "q_WCTA", "q_CLCA", "q_OENEG", "q_RETA", "q_SIZE", "q_NITA9", "q_OPTA9", "q_CFOTL9",
            "q_CHIN9", "q_REVG9", "q_LOSS9", "q_NITA_TTM"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(cover)


def main():
    rng = np.random.default_rng(42)
    _, pub, labels, mc, c2s = f7.load()
    r = pd.read_csv(ea.FULL / "classic_orig.csv", dtype={"corp_code": str})
    r["corp_code"] = r.corp_code.str.zfill(8)
    A, B, M = f7.design(r, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    A, B, M = A.reset_index(drop=True), B.reset_index(drop=True), M.reset_index(drop=True)
    Q, cover = q3_features(M, B)
    te = M.pred_year.isin(TEST).values
    cov = cover[te].groupby("pred_year")[["q3_있음", "T 이전 접수"]].mean().round(3)
    ev_cov = cover[te & (M.y.values == 1)][["q3_있음", "T 이전 접수"]].mean().round(3)
    print("3분기 자료 커버리지(채점 행):\n", cov.to_string(), "\n사건 행:", ev_cov.to_dict(), flush=True)

    X = {"A": A, "B": B, "B′": pd.concat([B, Q], axis=1), "A+Q": pd.concat([A, Q], axis=1),
         "Q만": Q}
    P = {k: f7.rolling(v, M, START, TEST) for k, v in X.items()}
    y = M.loc[te, "y"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)
    score = pd.DataFrame([{"조건": k, **{a: round(float(b), 4) for a, b in dm.score(y, p).items()}} for k, p in P.items()])
    cis = []
    for a, b, what in [("A", "B", "본문 누출"), ("A", "B′", "B′ 기준 누출"), ("A+Q", "B′", "같은 정보 집합 누출"),
                       ("B′", "B", "3분기 정보의 가치")]:
        for rr in ea.boot_ci(y, P[a], P[b], grp, 1000, rng):
            f_ = rr["0초과"]
            cis.append({"비교": f"{a} − {b}", "의미": what, **rr, "p(양측)": round(max(2 * min(f_, 1 - f_), 0.002), 3)})
    cis = pd.DataFrame(cis)
    share = {"남는 비율(A−B′)/(A−B)": round(float(cis.query("비교=='A − B′' and 지표=='PR-AUC'").격차.iloc[0]
                                              / cis.query("비교=='A − B' and 지표=='PR-AUC'").격차.iloc[0]), 3)}
    score.to_csv("results/final47_score.csv", index=False, encoding="utf-8-sig")
    cis.to_csv("results/final47_ci.csv", index=False, encoding="utf-8-sig")
    cov.to_csv("results/final47_cover.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    txt = "\n\n".join([cov.to_string(), "사건 행 커버리지: " + str(ev_cov.to_dict()), score.to_string(index=False),
                       cis[cis.지표.isin(["PR-AUC", "AUC", "상위100"])].to_string(index=False), str(share)])
    open("results/final47.log", "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
