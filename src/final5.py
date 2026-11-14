"""최종 실험 v2 — 후보 재구성 · 희소 변수 표 · 규정 경로 분해.

판정 기준은 하나다: T 이전에 공개됐는가(rcept_dt < T). 라벨과 가깝다는 이유로는 빼지 않는다.
대신 후보를 두 묶음으로 나눠 복구율을 분해한다.

    C_행동  정기보고서 지연·감사인 + 수시공시 빈도 + 정정 비율 + 공급계약 정정·해지 비율
            + 채무보증 + 금요일·연휴 전날 공시 비중 + 마감 직전 제출 비중
    C_규정  관리종목 지정으로 규정상 이어지는 공시 (L_ 계열 전체: 사유발생·환기·정지·
            실질심사·회생·불성실공시·소속부 변경)

선별(예측연도 2013~2017, 두 평가창 공통 학습 구간만 사용)
    1) B + C_행동으로 LightGBM → 학습 데이터 평균 |SHAP| 상위 10개
    2) 희소 변수 표: 켜진 표본 수·조건부 사건률·승산비(95% CI).
       켜진 표본 30건 이상이고 승산비 하한 > 1이면 SHAP 순위와 무관하게 포함
    C_규정은 전부 넣는다 (분해용 묶음이므로 거르지 않는다)

조건: A · A+ · A++ · B · B+ · B++   (+ = C_행동 선별 목록, ++ = + 와 C_규정)
출력: results/final5_*.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

import disclosure_mock as dm
import expand_analysis as ea
import feature_screen as fs
import gate3e_full as g
from final4 import BASE

warnings.filterwarnings("ignore")
K = 10
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1000
SEL_YEARS = list(range(2013, 2018))
MIN_ON, OUT = 30, "results/final5_"

# 채무보증을 담보제공에서 떼어낸다 — 계열사 우회 지원 경로라 따로 본다
CATS = [(c, p) for c, p in fs.CATS if c != "pledge"]
i = [c for c, _ in CATS].index("loan")
CATS[i:i] = [("guarantee", r"채무보증"), ("pledge", r"담보제공")]
fs.CATS = CATS
BEHAV = [c for c, _ in CATS if not c.startswith("L_") and c != "contract"]
RULE = [c for c, _ in CATS if c.startswith("L_")]


def window(a, T, t0):
    return np.searchsorted(a, T) - np.searchsorted(a, t0)


def extra_features(M, f, orig):
    """정정 비율 · 공급계약 정정·해지 비율 · 요일 · 마감 직전 제출."""
    beh = f[~f.cat.str.startswith("L_")].copy()
    # 영업일 달력 대용: 전체 공시가 한 건이라도 접수된 날 = 영업일
    days = np.sort(f.rcept_dt.dropna().unique())
    nxt = pd.Series(np.append(days[1:], days[-1]), index=days)
    gap = (nxt - nxt.index).dt.days
    beh["fri"] = beh.rcept_dt.dt.weekday == 4
    beh["pre_hol"] = beh.rcept_dt.map(gap).fillna(1).gt(3) | (
        beh.rcept_dt.map(gap).fillna(1).gt(1) & ~beh.fri)
    beh["amend"] = beh.report_nm.str.contains("정정")
    ct = beh[beh.cat == "contract"]
    ct_bad = ct[ct.report_nm.str.contains("정정|해지")]

    def arrs(d):
        return {c: np.sort(x.rcept_dt.values) for c, x in d.groupby("corp_code")}
    A_all, A_amd, A_fri, A_hol = arrs(beh), arrs(beh[beh.amend]), arrs(beh[beh.fri]), arrs(beh[beh.pre_hol])
    A_ct, A_ctb = arrs(ct[~ct.report_nm.str.contains("정정|해지")]), arrs(ct_bad)
    o = orig.copy()
    o["cram"] = (o.report_type.map(dm.DEADLINE) - o.lag).between(0, 3)
    by_o = {c: d.sort_values("rcept_dt") for c, d in o.groupby("corp_code")}

    E = []
    emp = np.array([], dtype="datetime64[ns]")
    for code, y in zip(M.corp_code, M.pred_year):
        T = np.datetime64(pd.Timestamp(year=y, month=1, day=1))
        t3 = np.datetime64(pd.Timestamp(year=y - 3, month=1, day=1))
        n = window(A_all.get(code, emp), T, t3)
        r = {"amend_share_3y": window(A_amd.get(code, emp), T, t3) / n if n else np.nan,
             "fri_share_3y": window(A_fri.get(code, emp), T, t3) / n if n else np.nan,
             "prehol_share_3y": window(A_hol.get(code, emp), T, t3) / n if n else np.nan}
        c_ok, c_bad = window(A_ct.get(code, emp), T, t3), window(A_ctb.get(code, emp), T, t3)
        r["contract_bad_share_3y"] = c_bad / c_ok if c_ok else np.nan
        d = by_o.get(code)
        if d is not None:
            seen = d[d.rcept_dt < pd.Timestamp(T)].tail(8)
            r["cram_share_2y"] = float(seen.cram.mean()) if len(seen) else np.nan
        E.append(r)
    return pd.DataFrame(E, index=M.index)


def sparse_table(C, M, cols):
    tr = M.pred_year.isin(SEL_YEARS).values
    y = M.loc[tr, "y"].values
    rows = []
    for c in cols:
        on = (C.loc[tr, c].fillna(0) > 0).values
        a, b = y[on].sum(), on.sum() - y[on].sum()
        cc, dd = y[~on].sum(), (~on).sum() - y[~on].sum()
        a_, b_, c_, d_ = a + .5, b + .5, cc + .5, dd + .5
        lor, se = np.log(a_ * d_ / (b_ * c_)), np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
        rows.append({"변수": c, "켜진표본": int(on.sum()), "켜진사건": int(a),
                     "사건률_켜짐": round(a / on.sum(), 3) if on.sum() else np.nan,
                     "사건률_꺼짐": round(cc / (~on).sum(), 3),
                     "승산비": round(np.exp(lor), 2), "하한": round(np.exp(lor - 1.96 * se), 2),
                     "상한": round(np.exp(lor + 1.96 * se), 2)})
    t = pd.DataFrame(rows).sort_values("승산비", ascending=False)
    t["강제포함"] = (t.켜진표본 >= MIN_ON) & (t.하한 > 1)
    return t


def main():
    rng = np.random.default_rng(42)
    ratios, pub, labels = ea.load_market("KOSDAQ", "classic")
    o = ratios
    ratios["O_SCORE"] = (-1.32 - 0.407 * o.O_SIZE + 6.03 * o.O_TLTA - 1.43 * o.O_WCTA
                         + 0.0757 * o.O_CLCA - 1.72 * o.O_OENEG - 2.37 * o.O_NITA
                         - 1.83 * o.O_FUTL_CFO + 0.285 * o.O_INTWO - 0.521 * o.O_CHIN)
    g.RATIO_COLS = BASE
    A, B, M = g.build(ratios, pub, labels, list(range(2013, 2025)), 1)
    print(f"행 {len(M):,} / 사건 {M.y.sum():,}", flush=True)

    orig, amend = dm.load_filings()
    aud = dm.load_auditors()
    f = fs.load_filings_all()
    f["cat"] = fs.classify(f["report_nm"])
    D = dm.features(M, 1, orig, amend, aud).drop(columns=["amend_1y", "amend_3y"])
    Eb = fs.event_features(M, 1, f, BEHAV).drop(columns=["amend_any_1y", "amend_any_3y", "amend_share_3y"])
    X5 = extra_features(M, f, orig)
    CB = pd.concat([D, Eb, X5], axis=1).reset_index(drop=True)
    CB = CB.loc[:, CB.notna().any() & (CB.nunique() > 1)]
    Er = fs.event_features(M, 1, f, RULE).reset_index(drop=True)
    CR = Er[[c for c in Er.columns if c.startswith("L_")]]
    CR = CR.loc[:, CR.nunique() > 1]
    print(f"C_행동 {CB.shape[1]}개 / C_규정 {CR.shape[1]}개", flush=True)

    import shap
    ea.g.TRAIN_YEARS, ea.g.TEST_YEARS = SEL_YEARS, list(range(2018, 2025))
    clf, Xtr, _, _ = fs.fit_predict(pd.concat([B, CB], axis=1), M)
    sv = shap.TreeExplainer(clf).shap_values(Xtr)
    sv = sv[1] if isinstance(sv, list) else sv
    rank = pd.Series(np.abs(sv).mean(0), index=Xtr.columns)[CB.columns].sort_values(ascending=False)
    rank.rename("mean_abs_shap").to_csv(OUT + "shap.csv", encoding="utf-8-sig")
    top = list(rank.index[:K])

    count_cols = [c for c in CB.columns if c.endswith(("_1y", "_3y")) and "share" not in c]
    sp = sparse_table(CB, M, count_cols)
    sp.to_csv(OUT + "sparse.csv", index=False, encoding="utf-8-sig")
    forced = [c for c in sp[sp.강제포함].변수 if c not in top]
    sel = top + forced
    print("SHAP 상위:", top, "\n강제포함:", forced, flush=True)
    print(sp.head(25).to_string(), flush=True)

    scores, cis = [], []
    for split in ("A", "B"):
        ea.set_split(split)
        te = M.pred_year.isin(g.TEST_YEARS).values
        grp = M.loc[te, "corp_code"].reset_index(drop=True)
        conds = {"A": A, "A+": pd.concat([A, CB[sel]], axis=1),
                 "A++": pd.concat([A, CB[sel], CR], axis=1),
                 "B": B, "B+": pd.concat([B, CB[sel]], axis=1),
                 "B++": pd.concat([B, CB[sel], CR], axis=1)}
        P = {}
        for name, X in conds.items():
            _, _, p, y = fs.fit_predict(X, M)
            P[name] = p
            scores.append({"평가창": split, "조건": name, "사건": int(y.sum()),
                           **{k: round(float(v), 4) for k, v in dm.score(y, p).items()}})
            print(scores[-1], flush=True)
        for a, b, what in [("A", "B", "누출 효과"), ("B+", "B", "행동 신호 기여"),
                           ("B++", "B+", "규정 경로 기여"), ("A", "B+", "남은 격차(행동)"),
                           ("A", "B++", "남은 격차(행동+규정)"), ("A+", "B+", "같은 변수 뒤 격차"),
                           ("A++", "B++", "같은 변수 뒤 격차(규정 포함)")]:
            for r in ea.boot_ci(y, P[a], P[b], grp, N_BOOT, rng):
                cis.append({"평가창": split, "비교": f"{a}−{b}", "의미": what, **r})

    S = pd.DataFrame(scores)
    S.to_csv(OUT + "score.csv", index=False, encoding="utf-8-sig")
    ci = pd.DataFrame(cis)
    ci.to_csv(OUT + "ci.csv", index=False, encoding="utf-8-sig")
    # 복구율 분해 (PR-AUC 기준)
    for split in ("A", "B"):
        s = S[S.평가창 == split].set_index("조건")["PR-AUC"]
        gap = s["A"] - s["B"]
        print(f"[창{split}] 누출 {gap:.3f} | 행동 복구 {(s['B+'] - s['B']) / gap:.0%} | "
              f"규정 추가 복구 {(s['B++'] - s['B+']) / gap:.0%}")
    print(ci.to_string())


if __name__ == "__main__":
    main()
