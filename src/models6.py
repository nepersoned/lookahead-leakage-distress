"""R1 — 복제 대상 논문의 모델 6종으로 A−B를 다시 잰다.

윤양현 외(2022) <표 5>·<표 12~22>의 명세를 그대로 쓴다.
목적은 성능 재현이 아니라 **"모델을 바꿔도 격차가 남는가"**다.
6종 중 5종 이상에서 같은 방향이어야 원인이 특정 알고리즘이 아님이 성립한다.

논문과 다른 점 — 논문은 1:1 쌍대표본에 무작위 80/20 분할이다. 우리는
불균형 전수에 시간순 분할을 쓴다. 그래서 절대 성능은 논문보다 낮게 나온다.
비교 대상은 논문의 82.73%가 아니라 **A와 B 사이의 차이**다.

사용: python src/models6.py --month 1 --vars 21
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import gate3e_full as g
import metrics as mx

warnings.filterwarnings("ignore")
OUT = mx.OUT


def build_models(seed: int = 42):
    """논문 <표 5>의 6종. 하이퍼파라미터는 <표 12>~<표 22>의 '최적' 조합.

    정규화 여부도 논문을 따른다 — LR·SVM·소프트보팅은 정규화, 트리 계열은 안 함.
    불균형 전수 표본이므로 class_weight="balanced"만 공통으로 더한다
    (논문은 1:1이라 필요가 없었다). 이건 A·B 양쪽에 똑같이 들어가므로
    차분에서 상쇄된다.
    """
    def norm(est):
        return make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler(), est)

    def raw(est):
        return make_pipeline(SimpleImputer(strategy="median"), est)

    lr = LogisticRegression(C=1, max_iter=10000, penalty=None, solver="sag",
                            class_weight="balanced")
    dt = DecisionTreeClassifier(max_depth=3, min_samples_split=2,
                                min_samples_leaf=2, class_weight="balanced",
                                random_state=seed)
    svm = SVC(probability=True, class_weight="balanced", random_state=seed)
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                random_state=seed, n_jobs=-1)
    import lightgbm as lgb
    gbm = lgb.LGBMClassifier(max_depth=5, n_estimators=500, num_leaves=6,
                             min_child_samples=10, subsample=0.5,
                             class_weight="balanced", verbose=-1,
                             random_state=seed)
    vote = VotingClassifier(
        estimators=[("lr", LogisticRegression(C=1, max_iter=10000, penalty=None,
                                              solver="sag",
                                              class_weight="balanced")),
                    ("rf", RandomForestClassifier(n_estimators=500,
                                                  class_weight="balanced",
                                                  random_state=seed, n_jobs=-1)),
                    ("svm", SVC(probability=True, class_weight="balanced",
                                random_state=seed))],
        voting="soft")
    return {
        "1 로지스틱": norm(lr),
        "2 의사결정나무": raw(dt),
        "3 SVM": norm(svm),
        "4 소프트보팅": norm(vote),
        "5 랜덤포레스트": raw(rf),
        "6 LightGBM": raw(gbm),
    }


def run_one(model, X, M):
    tr = M["pred_year"].isin(g.TRAIN_YEARS).values
    te = M["pred_year"].isin(g.TEST_YEARS).values
    lo, hi = X[tr].quantile(.01), X[tr].quantile(.99)
    model.fit(X[tr].clip(lo, hi, axis=1), M.loc[tr, "y"].values)
    p = model.predict_proba(X[te].clip(lo, hi, axis=1))[:, 1]
    return p, M.loc[te, "y"].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", type=int, default=1)
    ap.add_argument("--vars", default="21", choices=["10", "21"])
    a = ap.parse_args()

    ratios, cols = mx.load_ratios(a.vars)
    _, pub, labels = g.load()
    A, B, M = mx._build_generic(ratios, cols, pub, labels, a.month)
    print(f"관측 {len(M):,}행 · 평가 사건 "
          f"{int(M.loc[M.pred_year.isin(g.TEST_YEARS), 'y'].sum())}건 "
          f"· 변수 {len(cols)}개\n")

    rows = []
    for name in build_models():
        r = {"모델": name}
        for tag, X in (("A", A), ("B", B)):
            p, y = run_one(build_models()[name], X, M)
            pan = mx.panel(y, p)
            r[f"{tag} PR-AUC"] = round(pan["PR-AUC"], 3)
            r[f"{tag} 상위50"] = pan["상위50 적중"]
            r[f"{tag} AUC"] = round(pan["AUC"], 3)
        r["격차 PR-AUC"] = round(r["A PR-AUC"] - r["B PR-AUC"], 3)
        r["격차 상위50"] = r["A 상위50"] - r["B 상위50"]
        rows.append(r)
        print(f"  {name:14s} PR-AUC {r['A PR-AUC']:.3f} / {r['B PR-AUC']:.3f}"
              f"  격차 {r['격차 PR-AUC']:+.3f}"
              f"  |  상위50 {r['A 상위50']:2d} vs {r['B 상위50']:2d}", flush=True)

    df = pd.DataFrame(rows)
    same = int((df["격차 PR-AUC"] > 0).sum())
    print(f"\n6종 중 {same}종에서 A > B (PR-AUC 기준)")
    print(f"상위 50개 기준으로는 {int((df['격차 상위50'] > 0).sum())}종")
    name = f"models6_m{a.month}_v{a.vars}.csv"
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    print(f"\n→ results/{name}")


if __name__ == "__main__":
    main()
