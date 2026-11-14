"""최종본(v3) 숫자를 한 번에 만든다 — 원문판 재무 + 고친 모집단.

모집단 규칙(2026-10-02 수정, `gate3e_full.in_status`)
    T 시점에 관리종목 상태인 기업만 뺀다. 지정 이후 '관리종목 해제'·상장폐지 전이면 상태로 본다.
    이전 판은 "이전 지정 이력이 있고 올해 재지정되지 않은 기업"을 빼서, 라벨을 보고 표본을 골랐다.

재무: 최초 공시본 원문(`classic_orig.csv`). 영업현금흐름만 API 값.

단계
    1. 선별 — 2017~2018에서 B + 공시 후보로 SHAP 상위 10 + 희소 강제포함 (final7과 같은 규칙)
    2. 주 분석 — A · B · B+ · A+ · B+시장 · A+시장, 채점 2019~2024 확장창
       공시 변수 = SHAP로 고른 행동 신호(상위 10 + 희소 강제포함) + 지정 이력 3개(final28). 규정 경로 공시는 쓰지 않는다
    3. 연도별 PR-AUC
    4. 남는 격차 안정성 — 선별 개수 k = 5 · 15 · 20 (LightGBM 설정이 결정적이라 시드는 검정이 아니다)
    (5번 단계 없음 — 시장 변수는 주가·거래량 10개뿐이라 분해할 것이 없다)
    6. 오류 분해 — A+시장만 상위 100에 올린 사건의 사유
    7. 1:1 쌍대표본 — 200회, 무작위 80/20 · 시간순
    8. 시점 스윕 — 1~7월 A·B

사용: python src/final27.py [--boot 1000] [--skip-pair] [--skip-sweep]
출력: results/final27_*.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import final7 as f7  # 정규식 수정이 import 시 적용된다
import final5 as f5
import final8 as f8
import final10 as f10
import final20 as f20
import final21 as f21
import final28 as f28
import disclosure_mock as dm
import expand_analysis as ea
import feature_screen as fs
import gate3e_full as g
from dart import Dart

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"
OUT = "results/final27_"
START, TEST, SEL_YEARS, K = 2017, list(range(2019, 2025)), [2017, 2018], 10


def save(df, name):
    df.to_csv(OUT + name + ".csv", index=False, encoding="utf-8-sig")


def disclosure(M):
    orig, amend = dm.load_filings()
    aud = dm.load_auditors()
    f = fs.load_filings_all()
    f["cat"] = fs.classify(f["report_nm"])
    D = dm.features(M, 1, orig, amend, aud).drop(columns=["amend_1y", "amend_3y"])
    Eb = fs.event_features(M, 1, f, f5.BEHAV).drop(columns=["amend_any_1y", "amend_any_3y", "amend_share_3y"])
    CB = pd.concat([D, Eb, f5.extra_features(M, f, orig)], axis=1).reset_index(drop=True)
    CB = CB.loc[:, CB.notna().any() & (CB.nunique() > 1)]
    return CB


def select(B, CB, M):
    import shap
    g.TRAIN_YEARS, g.TEST_YEARS = SEL_YEARS, [TEST[0]]
    clf, Xtr, _, _ = fs.fit_predict(pd.concat([B, CB], axis=1), M)
    sv = shap.TreeExplainer(clf).shap_values(Xtr)
    sv = sv[1] if isinstance(sv, list) else sv
    rank = pd.Series(np.abs(sv).mean(0), index=Xtr.columns)[CB.columns].sort_values(ascending=False)
    save(rank.rename("mean_abs_shap").rename_axis("변수").reset_index(), "shap")
    top = list(rank.index[:K])
    f5.SEL_YEARS = SEL_YEARS
    counts = [c for c in CB.columns if c.endswith(("_1y", "_3y")) and "share" not in c]
    sp = f5.sparse_table(CB, M, counts)
    n_sel = int(M.pred_year.isin(SEL_YEARS).sum())
    sp["강제포함"] = (sp.켜진표본 >= 30) & (sp.켜진표본 <= .1 * n_sel) & (sp.하한 > 1)
    save(sp, "sparse")
    sel = top + [c for c in sp[sp.강제포함].변수 if c not in top]
    print("SHAP 상위:", top, "\n강제포함:", sel[K:], flush=True)
    return sel, list(rank.index), list(sp[sp.강제포함].변수)


def score_rows(P, y, tag=""):
    return [{"조건": k, **{a: round(float(b), 4) for a, b in dm.score(y, p).items()}, **({"비고": tag} if tag else {})}
            for k, p in P.items()]


def cis(y, P, grp, pairs, n, rng):
    out = []
    for a, b, what in pairs:
        for r in ea.boot_ci(y, P[a], P[b], grp, n, rng):
            out.append({"비교": f"{a} − {b}", "의미": what, **r})
    return pd.DataFrame(out)


def reason_groups(ev):
    """사건에 지정 사유와 사유군을 붙인다 (final10·final15와 같은 규칙)."""
    kind = pd.concat([pd.read_csv(FULL / f, dtype={"stock_code": str}) for f in
                      ("kind_events_kosdaq_old.csv", "kind_events.csv")], ignore_index=True)
    kind = kind[kind.event == "지정"].copy()
    kind["rcept_dt"] = pd.to_datetime(kind.rcept_dt.astype(str).str[:19], errors="coerce")
    import kind_codes
    kind = kind_codes.fill(kind)          # 외국 기업의 빈 종목코드를 회사명으로 채운다
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes.stock_code.astype(str).str.strip().str.zfill(6)
    cmap = codes[codes.stock_code.str.len() == 6].drop_duplicates("stock_code").set_index("stock_code")["corp_code"]
    kind["corp_code"] = kind.stock_code.map(cmap)
    kind = kind.dropna(subset=["corp_code"])
    kind["corp_code"] = kind.corp_code.astype(str).str.zfill(8)
    kind["pred_year"] = kind.rcept_dt.dt.year
    first = (kind.sort_values("rcept_dt").groupby(["corp_code", "pred_year"], as_index=False)
             .first()[["corp_code", "pred_year", "reason", "rcept_dt"]])
    ev = ev.merge(first, on=["corp_code", "pred_year"], how="left")
    sub = (pd.read_csv("results/final15_subreason.csv", dtype={"corp_code": str})
           .drop_duplicates(["corp_code", "year"]).rename(columns={"year": "pred_year"}))
    sub["corp_code"] = sub.corp_code.str.zfill(8)
    ev = ev.merge(sub[["corp_code", "pred_year", "sub"]], on=["corp_code", "pred_year"], how="left")
    rs = ev.reason.fillna("미상").astype(str)
    s = ev["sub"].fillna("").astype(str)
    grp = np.where(s.isin(["감사의견", "자본잠식", "보고서미제출"]), "감사·결산 확정",
          np.where(s == "실질심사", "거래소 재량",
          np.where(s == "회생파산", "법적 절차",
          np.where(rs.str.contains("감사|자본잠식|손실|매출액|미제출|사업보고서"), "감사·결산 확정",
          np.where(rs.str.contains("회생|파산"), "법적 절차",
          np.where(rs.str.contains("실질|적격성"), "거래소 재량", "기타·불명"))))))
    ev["사유"] = rs
    ev["하위사유"] = s
    ev["사유군"] = grp
    ev["month"] = ev.rcept_dt.dt.month
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--skip-pair", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(42)

    _, pub, labels, mc, c2s = f7.load()
    ratios = pd.read_csv(FULL / "classic_orig.csv", dtype={"corp_code": str})
    ratios["corp_code"] = ratios.corp_code.str.zfill(8)
    A, B, M = f7.design(ratios, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    lab = labels.groupby("corp_code")["event_dt"].apply(list).to_dict()
    M["prior"] = [int(any(d < pd.Timestamp(year=y, month=1, day=1) for d in lab.get(c, [])))
                  for c, y in zip(M.corp_code, M.pred_year)]
    te = M.pred_year.isin(TEST).values
    y = M.loc[te, "y"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)
    pr = M.loc[te, "prior"].values
    info = {"전체 행": len(M), "채점 행": int(te.sum()), "채점 사건": int(y.sum()),
            "사건률": round(float(y.mean()), 4),
            "이력 있는 채점 행": int(pr.sum()), "그중 사건": int((pr * y).sum()),
            "선별 구간 행": int(M.pred_year.isin(SEL_YEARS).sum()),
            "선별 구간 사건": int(M.loc[M.pred_year.isin(SEL_YEARS), "y"].sum())}
    print(info, flush=True)
    save(pd.DataFrame([info]), "sample")

    # 1. 선별
    CB = disclosure(M)
    sel, rank, forced = select(B, CB, M)

    # 시장
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes.stock_code.astype(str).str.strip().str.zfill(6)
    c2s2 = (codes[codes.stock_code.str.len() == 6].drop_duplicates("corp_code")
            .set_index("corp_code")["stock_code"].to_dict())
    F = f21.features(M, f21.load_market("ohlcv", range(2014, 2025)), None, c2s2).reset_index(drop=True)
    # 시장 = 주가·거래량 10개 (KRX Open API 일별매매정보. 공매도 잔고는 공식 API에 없어 쓰지 않는다)

    # 2. 주 분석
    H = f28.history(M, labels)
    bp = pd.concat([B, CB[sel], H], axis=1)
    ap_ = pd.concat([A, CB[sel], H], axis=1)
    conds = {"A": A, "B": B, "B+": bp, "A+": ap_,
             "B+시장": pd.concat([bp, F], axis=1), "A+시장": pd.concat([ap_, F], axis=1)}
    P = {}
    for k, X in conds.items():
        P[k] = f7.rolling(X, M, START, TEST)
        print(k, {a_: round(float(b_), 4) for a_, b_ in dm.score(y, P[k]).items()}, flush=True)
    save(pd.DataFrame(score_rows(P, y)), "score")
    # 채점 행별 예측값 — 경제적 의미 분석(final33)이 읽는다
    save(pd.concat([M.loc[te, ["corp_code", "pred_year", "y"]].reset_index(drop=True),
                    pd.DataFrame({k: p for k, p in P.items()})], axis=1), "preds")
    ci = cis(y, P, grp, [("A", "B", "누출"), ("B+", "B", "공시 복구"),
                         ("B+", "A", "B+가 A를 넘는가"), ("A+", "B+", "남는 격차 (공시까지)"),
                         ("B+시장", "B+", "시장 기여"), ("B+시장", "A", "B+시장이 A를 넘는가"),
                         ("A+시장", "B+시장", "남는 격차 (시장까지)")], a.boot, rng)
    save(ci, "ci")
    print(ci[ci.지표.isin(["PR-AUC", "AUC", "상위50"])].to_string(), flush=True)

    # 3. 연도별
    yr = M.loc[te, "pred_year"].values
    save(pd.DataFrame([{"연도": int(t), "조건": k, "사건": int(y[yr == t].sum()),
                        **{a_: round(float(b_), 4) for a_, b_ in dm.score(y[yr == t], p[yr == t]).items()}}
                       for t in TEST for k, p in P.items()]), "by_year")

    # 4. 안정성 — 선별 개수 k
    rows = []
    for k in (5, 15, 20):
        sk = rank[:k] + [c for c in forced if c not in rank[:k]]
        pa = f7.rolling(pd.concat([A, CB[sk], H, F], axis=1), M, START, TEST)
        pb = f7.rolling(pd.concat([B, CB[sk], H, F], axis=1), M, START, TEST)
        for r in ea.boot_ci(y, pa, pb, grp, a.boot, rng):
            rows.append({"k": k, "비교": "A+시장 − B+시장", "A": round(float(dm.score(y, pa)["PR-AUC"]), 4),
                         "B": round(float(dm.score(y, pb)["PR-AUC"]), 4), **r})
    st = pd.DataFrame(rows)
    save(st, "k")
    print(st[st.지표.isin(["PR-AUC", "AUC"])].to_string(), flush=True)

    # 6. 오류 분해
    Mt = M[te].reset_index(drop=True)
    out = Mt[["corp_code", "pred_year", "y", "lag", "prior"]].copy()
    for k in ("A+시장", "B+시장"):
        r, _ = f10.ranks(P[k], Mt.pred_year.values)
        out[f"rank_{k}"] = r.astype(int)
    ev = reason_groups(out[out.y == 1].copy())
    ev["A+만"] = (ev["rank_A+시장"] <= 100) & (ev["rank_B+시장"] > 100)
    save(ev, "events")
    gsum = ev.groupby("사유군").agg(사건=("y", "size"), A적중=("rank_A+시장", lambda s: float((s <= 100).mean())),
                                  B적중=("rank_B+시장", lambda s: float((s <= 100).mean())), A만=("A+만", "sum"),
                                  A순위중앙=("rank_A+시장", "median"), B순위중앙=("rank_B+시장", "median")).reset_index()
    save(gsum, "groups")
    print(gsum.to_string(), "\nA+만 사건 월:", ev[ev["A+만"]].month.value_counts().to_dict(),
          "\n전체 사건 월:", ev.month.value_counts().to_dict(), flush=True)

    # 7. 1:1
    if not a.skip_pair:
        pc = {"A": A, "B": B, "A+": ap_, "B+": bp, "A+시장": conds["A+시장"], "B+시장": conds["B+시장"]}
        y_all = M.y.values
        ev_i, no_i = np.where(y_all == 1)[0], np.where(y_all == 0)[0]
        prng = np.random.default_rng(42)
        rows = []
        for split in ("무작위 80/20", "시간순"):
            res = {k: [] for k in pc}
            for rep in range(200):
                pick = np.concatenate([ev_i, prng.choice(no_i, len(ev_i), replace=False)])
                prng.shuffle(pick)
                yy = y_all[pick]
                if split == "무작위 80/20":
                    tr = np.zeros(len(pick), bool); tr[:int(len(pick) * .8)] = True
                else:
                    tr = M.pred_year.values[pick] <= 2022
                if yy[~tr].sum() < 5 or yy[tr].sum() < 5:
                    continue
                for k, X in pc.items():
                    res[k].append(f20.fit_eval(X.iloc[pick].reset_index(drop=True), yy, tr, ~tr, rep))
            for k, v in res.items():
                d = pd.DataFrame(v)
                rows.append({"분할": split, "조건": k, "반복": len(d), **d.mean().round(4).to_dict()})
        pr_ = pd.DataFrame(rows)
        save(pr_, "pair")
        print(pr_.to_string(), flush=True)

    # 8. 시점 스윕
    if not a.skip_sweep:
        rows, cs = [], []
        for month in range(1, 8):
            Am, Bm, Mm = f7.design(ratios, pub, labels, mc, c2s, list(range(2017, 2025)), month)
            tm = Mm.pred_year.isin(TEST).values
            ym = Mm.loc[tm, "y"].values
            pa, pb = f7.rolling(Am, Mm, START, TEST), f7.rolling(Bm, Mm, START, TEST)
            sa, sb = dm.score(ym, pa), dm.score(ym, pb)
            rows.append({"월": month, "행": int(tm.sum()), "사건": int(ym.sum()),
                         "lag>0 비율": round(float((Mm.loc[tm, "lag"] > 0).mean()), 4),
                         **{f"A_{k}": round(float(v), 4) for k, v in sa.items()},
                         **{f"B_{k}": round(float(v), 4) for k, v in sb.items()}})
            for r in ea.boot_ci(ym, pa, pb, Mm.loc[tm, "corp_code"].reset_index(drop=True), 500, rng):
                cs.append({"월": month, **r})
            print(rows[-1], flush=True)
        save(pd.DataFrame(rows), "sweep")
        save(pd.DataFrame(cs), "sweep_ci")


if __name__ == "__main__":
    main()
