"""공시 행태 변수 목업 — 정직한 모형(B)에 더하면 격차를 얼마나 메우나.

1부의 결론을 뒤집은 가설이다. 관행(A)은 예측 시점에 아직 없던 재무제표를 썼다.
그런데 "그 서류가 아직 없다", "늘 늦게 낸다", "자꾸 고친다"는 사실은 그 시점에
관측할 수 있다. DART 접수일이 있어서 한국에서만 만들 수 있는 변수들이다.

    ① 사업보고서 지연   최근 공개된 사업보고서의 결산일 대비 제출일수 · 기한 초과 · 전년 대비 변화
    ② 정기보고서 지연   최근 정기보고서 지연일수 · 최근 4건 중 기한 초과 비율
    ③ 정정 빈도         최근 1년·3년 정정공시 건수
    ④ 감사인            감사인 교체 · 비적정 의견 · 강조사항 (auditors_full.csv가 있을 때)

시점 규칙 — 모든 변수는 rcept_dt < T 인 공시만으로 만든다. "보고서 미제출"은 그 자체가
지정 사유이므로 T 이후의 미제출을 쓰면 정답을 읽는 셈이 된다.

회복률 = (C − B) / (A − B). A는 예측 시점에 만들 수 없는 모형이므로 100%를 넘을 필요는 없다.

사용: python src/disclosure_mock.py [--split A B] [--boot 1000]
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import expand_analysis as ea
import gate3e_full as g
import metrics as mx

warnings.filterwarnings("ignore")
FULL, OUT = ea.FULL, ea.OUT
DEADLINE = {"사업": 90, "반기": 45, "분기": 45}
NONCLEAN = ("한정", "부적정", "의견거절")
EMPTY_EMPH = {"", "-", "해당사항 없음", "해당사항없음", "없음", "N/A"}


def load_filings():
    rep = pd.read_csv(FULL / "reports_full.csv", dtype={"corp_code": str})
    rep["corp_code"] = rep["corp_code"].str.zfill(8)
    rep["rcept_dt"] = pd.to_datetime(rep["rcept_dt"].astype(str).str[:10], errors="coerce")
    rep["is_amended"] = rep["is_amended"].astype(str).str.lower().isin(["true", "1"])
    rep["period_dt"] = (pd.to_datetime(dict(year=rep.bsns_year, month=rep.period_end, day=1))
                        + pd.offsets.MonthEnd(0))
    # 지연은 원 제출본 기준. 같은 보고서가 여러 줄이면 가장 이른 원본을 쓴다
    orig = (rep[~rep.is_amended].sort_values("rcept_dt")
            .drop_duplicates(["corp_code", "report_type", "bsns_year", "period_end"]))
    orig["lag"] = (orig.rcept_dt - orig.period_dt).dt.days
    orig["late"] = orig.lag > orig.report_type.map(DEADLINE)
    amend = rep[rep.is_amended][["corp_code", "rcept_dt"]]
    return orig, amend


def load_auditors():
    p = FULL / "auditors_full.csv"
    if not p.exists():
        return None
    a = pd.read_csv(p, dtype={"corp_code": str}).fillna("")
    a["corp_code"] = a["corp_code"].str.zfill(8)
    # 표기 흔들림(띄어쓰기·유한·(주)·영문 병기)을 교체로 잘못 세지 않게 이름을 정리한다
    a["adtor"] = (a.adtor.str.replace(r"\s|\(.*?\)|유한|주식회사|회계법인|Deloitte|KPMG|EY|PwC",
                                      "", regex=True).str.strip())
    a = a.sort_values(["corp_code", "bsns_year"])
    a["prev_adtor"] = a.groupby("corp_code")["adtor"].shift()
    a["adtor_change"] = ((a.prev_adtor != "") & (a.adtor != "") & (a.prev_adtor != a.adtor)
                         & a.prev_adtor.notna()).astype(float)
    a["nonclean"] = a.adt_opinion.str.replace(r"\s", "", regex=True).str.contains(
        "|".join(NONCLEAN)).astype(float)
    a["emphasis"] = (~a.emphs_matter.str.strip().isin(EMPTY_EMPH)).astype(float)
    return a.set_index(["corp_code", "bsns_year"])[["adtor_change", "nonclean", "emphasis"]]


def features(M, month, orig, amend, aud):
    by_o = {c: d for c, d in orig.groupby("corp_code")}
    by_a = {c: d.rcept_dt.values for c, d in amend.groupby("corp_code")}
    rows = []
    for code, y in zip(M.corp_code, M.pred_year):
        T = pd.Timestamp(year=y, month=month, day=1)
        f = {}
        o = by_o.get(code)
        if o is not None:
            seen = o[o.rcept_dt < T]
            ann = seen[seen.report_type == "사업"].sort_values("bsns_year")
            if len(ann):
                f["ann_lag"] = ann.lag.iloc[-1]
                f["ann_late"] = float(ann.late.iloc[-1])
                if len(ann) > 1:
                    f["ann_lag_chg"] = ann.lag.iloc[-1] - ann.lag.iloc[-2]
                fy_b = int(ann.bsns_year.iloc[-1])
                if aud is not None and (code, fy_b) in aud.index:
                    f.update(aud.loc[(code, fy_b)].to_dict())
            last4 = seen.sort_values("rcept_dt").tail(4)
            if len(last4):
                f["q_lag"] = last4.lag.iloc[-1]
                f["q_late_share"] = float(last4.late.mean())
                f["q_lag_max"] = float(last4.lag.max())
        am = by_a.get(code)
        if am is not None:
            f["amend_1y"] = float(((am < np.datetime64(T)) &
                                   (am >= np.datetime64(T - pd.DateOffset(years=1)))).sum())
            f["amend_3y"] = float(((am < np.datetime64(T)) &
                                   (am >= np.datetime64(T - pd.DateOffset(years=3)))).sum())
        else:
            f["amend_1y"] = f["amend_3y"] = 0.0
        rows.append(f)
    return pd.DataFrame(rows, index=M.index)


GROUPS = {"① 사업보고서 지연": ["ann_lag", "ann_late", "ann_lag_chg"],
          "② 정기보고서 지연": ["q_lag", "q_late_share", "q_lag_max"],
          "③ 정정 빈도": ["amend_1y", "amend_3y"],
          "④ 감사인": ["adtor_change", "nonclean", "emphasis"]}


def score(y, p):
    o = np.argsort(-p)
    return {"PR-AUC": average_precision_score(y, p), "AUC": roc_auc_score(y, p),
            "상위50": int(y[o[:50]].sum()), "상위100": int(y[o[:100]].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["A", "B"])
    ap.add_argument("--month", type=int, default=1)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    rng = np.random.default_rng(42)

    g.RATIO_COLS = mx.VARS21
    ratios, pub, labels = ea.load_market("KOSDAQ", "21")
    orig, amend = load_filings()
    aud = load_auditors()
    groups = {k: v for k, v in GROUPS.items() if aud is not None or k != "④ 감사인"}
    if aud is None:
        print("auditors_full.csv 없음 — ④ 감사인은 빼고 돈다")

    allrows = []
    for split in a.split:
        ea.set_split(split)
        A, B, M = g.build(ratios, pub, labels, g.TRAIN_YEARS + g.TEST_YEARS, a.month)
        F = features(M, a.month, orig, amend, aud)
        te = M.pred_year.isin(g.TEST_YEARS).values
        mt = M[te].reset_index(drop=True)

        conds = {"A 관행": A, "B 정직": B}
        for name, cols in groups.items():
            conds[f"B + {name}"] = pd.concat([B, F.reindex(columns=cols)], axis=1)
        conds["C = B + 공시행태 전부"] = pd.concat(
            [B, F.reindex(columns=sum(groups.values(), []))], axis=1)

        preds = {}
        for name, X in conds.items():
            p, y = mx._fit(X.reset_index(drop=True), M.reset_index(drop=True))
            preds[name] = p
        yt = mt.y.values
        sa, sb = score(yt, preds["A 관행"]), score(yt, preds["B 정직"])

        print(f"\n### 코스닥 · 평가창 {split} · {a.month}월 — 평가 {len(yt):,}행 · 사건 {int(yt.sum())}")
        for name, p in preds.items():
            s = score(yt, p)
            row = {"split": split, "조건": name, **{k: round(v, 4) if isinstance(v, float) else v
                                                   for k, v in s.items()}}
            for k in ("PR-AUC", "AUC", "상위50"):
                gap = sa[k] - sb[k]
                row[f"회복률 {k}"] = round((s[k] - sb[k]) / gap, 3) if gap else None
            allrows.append(row)

        # C − B 신뢰구간 (기업 단위)
        pc, pb = preds["C = B + 공시행태 전부"], preds["B 정직"]
        groups_idx = pd.Series(range(len(mt))).groupby(mt.corp_code.values).indices
        keys = list(groups_idx)
        draws = {"PR-AUC": [], "AUC": [], "상위50": []}
        for _ in range(a.boot):
            pick = rng.integers(0, len(keys), len(keys))
            idx = np.concatenate([groups_idx[keys[j]] for j in pick])
            if yt[idx].sum() < 5:
                continue
            sc, sbb = score(yt[idx], pc[idx]), score(yt[idx], pb[idx])
            for k in draws:
                draws[k].append(sc[k] - sbb[k])
        ci = {k: np.percentile(v, [2.5, 97.5]).round(4).tolist() for k, v in draws.items()}

        df = pd.DataFrame([r for r in allrows if r["split"] == split])
        show = ["조건", "PR-AUC", "AUC", "상위50", "상위100", "회복률 PR-AUC", "회복률 상위50"]
        print(df[show].to_string(index=False))
        print(f"C − B 95% 구간: {ci}")

        # 변수 중요도 (LightGBM gain, 평가창 학습분)
        import lightgbm as lgb
        X = conds["C = B + 공시행태 전부"].reset_index(drop=True)
        Mm = M.reset_index(drop=True)
        tr = Mm.pred_year.isin(g.TRAIN_YEARS).values
        lo, hi = X[tr].quantile(.01), X[tr].quantile(.99)
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15,
                                 min_child_samples=10, class_weight="balanced",
                                 verbose=-1, random_state=42, importance_type="gain")
        clf.fit(X[tr].clip(lo, hi, axis=1), Mm.loc[tr, "y"].values)
        imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
        imp = (imp / imp.sum()).round(3)
        print("중요도 상위 10:", imp.head(10).to_dict())
        imp.to_csv(OUT / f"disclosure_importance_{split}.csv", encoding="utf-8-sig")

        print("\n[공시행태 변수 — 사건기업 vs 정상기업 평균 (평가창 학습분)]")
        F2 = F.reset_index(drop=True)
        cmp_ = pd.DataFrame({"사건": F2[Mm.y == 1].mean(), "정상": F2[Mm.y == 0].mean(),
                             "결측률": F2.isna().mean()}).round(3)
        print(cmp_.to_string())

    pd.DataFrame(allrows).to_csv(OUT / "disclosure_mock.csv", index=False, encoding="utf-8-sig")
    print("\n→ results/disclosure_mock.csv")


if __name__ == "__main__":
    main()
