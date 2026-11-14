"""리픽싱·담보 공시를 범주로 분리하고 다시 선별·평가한다.

발견: 전환가액·행사가액·교환가액 조정 공시 10,762건 중 **8,881건이 `other`로 버려지고
1,856건이 `bw`에 섞여** 있었다. 리픽싱은 주가가 기준 아래로 내려갔다는 자동 신호라
따로 세야 한다. `최대주주변경을수반하는주식담보제공`(3,426건)도 단순 최대주주 변경과
성격이 다르다.

추가 범주
    refix         전환가액·신주인수권행사가액·교환가액의 조정
    owner_pledge  최대주주 변경을 수반하는 주식 담보제공 (계약 체결·해제)
    owner         나머지 최대주주 변경

선별과 평가는 본문(final7/final8)과 같다. 선별을 다시 하는 이유는 후보가 바뀌었기
때문이며, 선별 구간(2017~2018)은 그대로다.

출력: results/final16_*.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import final7 as f7            # 정규식 순서 수정(embezzle/acquire)이 import 시 적용된다
import final5 as f5
import final8 as f8
import disclosure_mock as dm
import expand_analysis as ea
import feature_screen as fs
import gate3e_full as g
from final4 import BASE

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TEST, START, K = f8.TEST, f8.START, 10

# ── 범주 추가: refix는 cb/bw보다 앞에 둬야 '신주인수권행사가액의 조정'을 먼저 잡는다
CATS = [(c, p) for c, p in f5.CATS]
names = [c for c, _ in CATS]
CATS.insert(names.index("cb"), ("refix", r"(전환가액|행사가액|교환가액)\s*의?\s*조정"))
names = [c for c, _ in CATS]
CATS.insert(names.index("owner"), ("owner_pledge", r"최대주주\s*변경.*담보제공"))
fs.CATS = CATS
f5.CATS = CATS
f5.BEHAV = [c for c, _ in CATS if not c.startswith("L_") and c != "contract"]
f5.RULE = [c for c, _ in CATS if c.startswith("L_")]


def main():
    rng = np.random.default_rng(42)
    ratios, pub, labels, mc, c2s = f7.load()
    A, B, M = f7.design(ratios, pub, labels, mc, c2s, list(range(2017, 2025)), 1)

    orig, amend = dm.load_filings()
    aud = dm.load_auditors()
    f = fs.load_filings_all()
    f["cat"] = fs.classify(f["report_nm"])
    print("새 범주 건수:", f.cat.value_counts().reindex(
        ["refix", "owner_pledge", "owner", "cb", "bw", "other"]).to_dict(), flush=True)

    D = dm.features(M, 1, orig, amend, aud).drop(columns=["amend_1y", "amend_3y"])
    Eb = fs.event_features(M, 1, f, f5.BEHAV).drop(
        columns=["amend_any_1y", "amend_any_3y", "amend_share_3y"])
    CB = pd.concat([D, Eb, f5.extra_features(M, f, orig)], axis=1).reset_index(drop=True)
    CB = CB.loc[:, CB.notna().any() & (CB.nunique() > 1)]
    Er = fs.event_features(M, 1, f, f5.RULE).reset_index(drop=True)
    CR = Er[[c for c in Er.columns if c.startswith("L_")]]
    CR = CR.loc[:, CR.nunique() > 1]
    print(f"C_행동 {CB.shape[1]}개 / C_규정 {CR.shape[1]}개", flush=True)

    # ── 선별 (2017~2018, 학습 데이터 SHAP + 희소 변수 기준) — 본문과 같은 규칙
    import shap
    sel_years = [2017, 2018]
    g.TRAIN_YEARS, g.TEST_YEARS = sel_years, [TEST[0]]
    clf, Xtr, _, _ = fs.fit_predict(pd.concat([B, CB], axis=1), M)
    sv = shap.TreeExplainer(clf).shap_values(Xtr)
    sv = sv[1] if isinstance(sv, list) else sv
    rank = pd.Series(np.abs(sv).mean(0), index=Xtr.columns)[CB.columns].sort_values(ascending=False)
    rank.rename("mean_abs_shap").to_csv(RES / "final16_shap.csv", encoding="utf-8-sig")
    top = list(rank.index[:K])
    f5.SEL_YEARS = sel_years
    counts = [c for c in CB.columns if c.endswith(("_1y", "_3y")) and "share" not in c]
    sp = f5.sparse_table(CB, M, counts)
    n_sel = int(M.pred_year.isin(sel_years).sum())
    sp["강제포함"] = (sp.켜진표본 >= 30) & (sp.켜진표본 <= .1 * n_sel) & (sp.하한 > 1)
    sp.to_csv(RES / "final16_sparse.csv", index=False, encoding="utf-8-sig")
    sel = top + [c for c in sp[sp.강제포함].변수 if c not in top]
    print("SHAP 상위:", top, "\n강제포함:", sel[K:], flush=True)
    print("\n새 변수의 자리\n" + rank.reset_index().rename(
        columns={"index": "변수", 0: "shap"}).assign(순위=lambda d: d.index + 1)
        .query("변수.str.startswith('refix') or 변수.str.startswith('owner')").to_string(index=False))
    print("\n새 변수 승산비\n" + sp[sp.변수.str.startswith(("refix", "owner"))].to_string(index=False))

    # ── 평가
    old_sel = f8.selected()
    conds = {"A": A, "B": B,
             "A+ (기존)": pd.concat([A, CB[[c for c in old_sel if c in CB]], CR], axis=1),
             "B+ (기존)": pd.concat([B, CB[[c for c in old_sel if c in CB]], CR], axis=1),
             "A+ (리픽싱 포함)": pd.concat([A, CB[sel], CR], axis=1),
             "B+ (리픽싱 포함)": pd.concat([B, CB[sel], CR], axis=1)}
    te = M.pred_year.isin(TEST).values
    y = M.loc[te, "y"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)
    P, rows = {}, []
    for name, X in conds.items():
        P[name] = f7.rolling(X, M, START, TEST)
        rows.append({"조건": name, "변수": X.shape[1],
                     **{k: round(float(v), 4) for k, v in dm.score(y, P[name]).items()}})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "final16_score.csv", index=False, encoding="utf-8-sig")

    cis = []
    for a, b, what in [("B+ (리픽싱 포함)", "B+ (기존)", "리픽싱 분리 효과"),
                       ("B+ (리픽싱 포함)", "B", "공시변수 기여"),
                       ("A", "B", "누출"),
                       ("A+ (리픽싱 포함)", "B+ (리픽싱 포함)", "남는 격차"),
                       ("B+ (리픽싱 포함)", "A", "B+가 관행을 넘는가")]:
        for r in ea.boot_ci(y, P[a], P[b], grp, 1000, rng):
            cis.append({"비교": f"{a} − {b}", "의미": what, **r})
    ci = pd.DataFrame(cis)
    ci.to_csv(RES / "final16_ci.csv", index=False, encoding="utf-8-sig")
    print(ci.to_string())


if __name__ == "__main__":
    main()
