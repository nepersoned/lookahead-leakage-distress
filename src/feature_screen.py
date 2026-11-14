"""공시 행태 후보 변수 선별 — 전부 넣고 SHAP으로 추린다.

    기준 모형   B (정직, 변수 21개). 고전 모형 변수는 넣지 않는다
    후보 변수   정기보고서 지연 · 정정 · 감사인 (disclosure_mock) + 수시공시 범주별 빈도
    선별        학습 구간에서만 SHAP 평균 |값|을 계산해 순위를 매긴다
    채점        고른 변수로 평가 구간을 한 번만 채점한다

선별에 평가 구간을 쓰면 고른 변수가 평가에서 좋게 나오는 게 당연해진다. 이 프로젝트가
지적하는 누출을 변수 선택 단계에서 저지르는 셈이므로 학습 구간만 쓴다.

"정답을 읽는" 공시는 따로 둔다 — 관리종목 사유 발생 자진공시, 투자주의환기종목,
매매거래정지, 상장적격성 실질심사, 회생·부도. 기본 후보에서 빼고 --with-label로만 넣는다.

모든 변수는 rcept_dt < T 인 공시만으로 만든다.

사용: python src/feature_screen.py [--top 10] [--boot 1000] [--with-label]
"""
from __future__ import annotations

import argparse
import re
import warnings

import numpy as np
import pandas as pd

import disclosure_mock as dm
import expand_analysis as ea
import gate3e_full as g
import metrics as mx

warnings.filterwarnings("ignore")
FULL, OUT = ea.FULL, ea.OUT

# 제목 → 범주. 위에서부터 먼저 걸리는 것으로 분류한다
CATS = [
    # --- 정답에 가까운 공시 (기본 제외)
    ("L_self_reason", r"사유\s*발생|지정\s*우려|폐지\s*우려|자본잠식률|손실발생|매출액미달|감사의견.*(거절|한정|부적정)"),
    ("L_warning", r"투자주의환기|투자경고|투자위험"),
    ("L_halt", r"매매거래정지|거래정지"),
    ("L_review", r"상장적격성|실질심사|기업심사위원회|개선기간|개선계획|상장폐지"),
    ("L_distress", r"회생절차|파산|부도|은행거래정지|영업정지"),
    ("L_unfaithful", r"불성실공시"),
    ("L_section", r"소속부\s*변경|기타시장안내"),
    # --- 후보 변수
    ("cb", r"전환사채"),
    ("bw", r"신주인수권부사채|신주인수권"),
    ("eb", r"교환사채"),
    ("rights", r"유상증자"),
    ("bonus", r"무상증자"),
    ("capred", r"감자"),
    ("owner", r"최대주주\s*변경|최대주주변경"),
    ("ceo", r"대표이사|대표집행임원"),
    ("embezzle", r"횡령|배임"),
    ("lawsuit", r"소송"),
    ("inquiry", r"조회공시|풍문|보도"),
    ("asset_sale", r"(유형자산|영업|자산)\s*양도"),
    ("acquire", r"(타법인주식|출자증권|영업|유형자산)\s*.*양수|합병|분할"),
    ("contract", r"공급계약|판매.*계약"),
    ("buyback", r"자기주식"),
    ("small_offer", r"소액공모"),
    ("pledge", r"담보제공|채무보증"),
    ("loan", r"금전대여|단기차입금증가"),
    ("gm", r"주주총회"),
]
LABEL_CATS = [c for c, _ in CATS if c.startswith("L_")]


def classify(names: pd.Series) -> pd.Series:
    clean = names.str.replace(r"\[[^\]]*\]", "", regex=True)
    out = pd.Series("other", index=names.index)
    for cat, pat in reversed(CATS):
        out[clean.str.contains(pat, regex=True)] = cat
    return out


def load_filings_all():
    f = pd.read_csv(FULL / "filings_all.csv", dtype={"corp_code": str, "rcept_no": str})
    f["corp_code"] = f["corp_code"].str.zfill(8)
    f["rcept_dt"] = pd.to_datetime(f["rcept_dt"].astype(str), format="%Y%m%d", errors="coerce")
    f["cat"] = classify(f["report_nm"])
    f["amended"] = f["report_nm"].str.contains(r"정정")
    return f


def event_features(M, month, f, cats):
    by = {}
    for (code, cat), d in f.groupby(["corp_code", "cat"]):
        by[(code, cat)] = np.sort(d.rcept_dt.values)
    # 총량 변수에도 정답 근처 공시가 섞이지 않게 한다 (첫 실행에서 이게 새서 결과가 부풀었다)
    base = f if any(c.startswith("L_") for c in cats) else f[~f.cat.str.startswith("L_")]
    tot = {c: np.sort(d.rcept_dt.values) for c, d in base.groupby("corp_code")}
    amd = {c: np.sort(d.rcept_dt.values) for c, d in base[base.amended].groupby("corp_code")}
    rows = []
    for code, y in zip(M.corp_code, M.pred_year):
        T = np.datetime64(pd.Timestamp(year=y, month=month, day=1))
        t1 = np.datetime64(pd.Timestamp(year=y - 1, month=month, day=1))
        t3 = np.datetime64(pd.Timestamp(year=y - 3, month=month, day=1))
        r = {}
        for cat in cats:
            a = by.get((code, cat))
            if a is None:
                r[f"{cat}_1y"] = r[f"{cat}_3y"] = 0.0
                continue
            hi = np.searchsorted(a, T)
            r[f"{cat}_1y"] = float(hi - np.searchsorted(a, t1))
            r[f"{cat}_3y"] = float(hi - np.searchsorted(a, t3))
        for key, src in (("all_filings", tot), ("amend_any", amd)):
            a = src.get(code)
            if a is None:
                r[f"{key}_1y"] = r[f"{key}_3y"] = 0.0
            else:
                hi = np.searchsorted(a, T)
                r[f"{key}_1y"] = float(hi - np.searchsorted(a, t1))
                r[f"{key}_3y"] = float(hi - np.searchsorted(a, t3))
        rows.append(r)
    E = pd.DataFrame(rows, index=M.index)
    E["amend_share_3y"] = E["amend_any_3y"] / E["all_filings_3y"].replace(0, np.nan)
    return E


def fit_predict(X, M, seed=42):
    import lightgbm as lgb
    tr = M.pred_year.isin(g.TRAIN_YEARS).values
    te = M.pred_year.isin(g.TEST_YEARS).values
    lo, hi = X[tr].quantile(.01), X[tr].quantile(.99)
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15,
                             min_child_samples=10, class_weight="balanced",
                             verbose=-1, random_state=seed)
    Xtr = X[tr].clip(lo, hi, axis=1)
    clf.fit(Xtr, M.loc[tr, "y"].values)
    return clf, Xtr, clf.predict_proba(X[te].clip(lo, hi, axis=1))[:, 1], M.loc[te, "y"].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["A", "B"])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--with-label", action="store_true")
    a = ap.parse_args()
    import shap
    rng = np.random.default_rng(42)

    g.RATIO_COLS = mx.VARS21
    ratios, pub, labels = ea.load_market("KOSDAQ", "21")
    orig, amend = dm.load_filings()
    aud = dm.load_auditors()
    f = load_filings_all()
    print("[수시공시 범주별 건수]")
    print(f.cat.value_counts().to_string())
    cats = [c for c, _ in CATS if a.with_label or not c.startswith("L_")]

    summary = []
    for split in a.split:
        ea.set_split(split)
        A, B, M = g.build(ratios, pub, labels, g.TRAIN_YEARS + g.TEST_YEARS, 1)
        D = dm.features(M, 1, orig, amend, aud)
        E = event_features(M, 1, f, cats)
        cand = pd.concat([D, E], axis=1)
        cand = cand.loc[:, cand.notna().mean() > 0.5]
        cand = cand.loc[:, cand.std() > 0]
        Mr = M.reset_index(drop=True)
        Br, Ar, Cr = B.reset_index(drop=True), A.reset_index(drop=True), cand.reset_index(drop=True)

        # 1) B + 후보 전부 → 학습 구간 SHAP
        Xall = pd.concat([Br, Cr], axis=1)
        clf, Xtr, p_all, y = fit_predict(Xall, Mr)
        sv = shap.TreeExplainer(clf).shap_values(Xtr)
        sv = sv[1] if isinstance(sv, list) else sv
        imp = pd.Series(np.abs(sv).mean(0), index=Xall.columns)
        cand_rank = imp[Cr.columns].sort_values(ascending=False)
        fin_rank = imp[Br.columns].sort_values(ascending=False)

        tr = Mr.pred_year.isin(g.TRAIN_YEARS).values
        desc = pd.DataFrame({
            "SHAP": cand_rank.round(4),
            "순위(전체)": imp.rank(ascending=False)[cand_rank.index].astype(int),
            "사건평균": Cr[tr & (Mr.y == 1)][cand_rank.index].mean().round(3),
            "정상평균": Cr[tr & (Mr.y == 0)][cand_rank.index].mean().round(3),
        })
        desc["배수"] = (desc.사건평균 / desc.정상평균.replace(0, np.nan)).round(2)
        desc.to_csv(OUT / f"screen_rank_{split}.csv", encoding="utf-8-sig")
        print(f"\n### 평가창 {split} — 학습 구간 SHAP 상위 후보 (재무비율 1위 {fin_rank.index[0]} = {fin_rank.iloc[0]:.4f})")
        print(desc.head(20).to_string())

        # 2) 상위 k개만 → 평가 구간 채점
        top = list(cand_rank.index[:a.top])
        preds = {}
        for name, X in [("A 관행", Ar), ("B 정직", Br),
                        (f"B+ 상위{a.top}", pd.concat([Br, Cr[top]], axis=1)),
                        ("B+ 후보 전부", Xall)]:
            _, _, p, yy = fit_predict(X, Mr)
            preds[name] = p
        yt = yy
        mt = Mr[Mr.pred_year.isin(g.TEST_YEARS)].reset_index(drop=True)
        sa, sb = dm.score(yt, preds["A 관행"]), dm.score(yt, preds["B 정직"])
        print(f"\n평가 {len(yt):,}행 · 사건 {int(yt.sum())}")
        for name, p in preds.items():
            s = dm.score(yt, p)
            row = {"split": split, "조건": name, **{k: round(v, 4) if isinstance(v, float) else v
                                                   for k, v in s.items()}}
            for k in ("PR-AUC", "AUC", "상위50"):
                gap = sa[k] - sb[k]
                row[f"회복률 {k}"] = round((s[k] - sb[k]) / gap, 3) if gap else None
            summary.append(row)
            print("  ", row)

        bp = preds[f"B+ 상위{a.top}"]
        gi = pd.Series(range(len(mt))).groupby(mt.corp_code.values).indices
        keys = list(gi)
        draws = {"PR-AUC": [], "AUC": [], "상위50": []}
        for _ in range(a.boot):
            pick = rng.integers(0, len(keys), len(keys))
            idx = np.concatenate([gi[keys[j]] for j in pick])
            if yt[idx].sum() < 5:
                continue
            s1, s0 = dm.score(yt[idx], bp[idx]), dm.score(yt[idx], preds["B 정직"][idx])
            for k in draws:
                draws[k].append(s1[k] - s0[k])
        print("  B+ − B 95% 구간:", {k: np.percentile(v, [2.5, 97.5]).round(4).tolist()
                                    for k, v in draws.items()})
        print("  고른 변수:", top)

    tag = "_label" if a.with_label else ""
    pd.DataFrame(summary).to_csv(OUT / f"screen_eval{tag}.csv", index=False, encoding="utf-8-sig")
    print("\n→ results/screen_rank_*.csv, screen_eval.csv")


if __name__ == "__main__":
    main()
