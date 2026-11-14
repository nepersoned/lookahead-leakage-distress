"""1:1 쌍대표본에서 지금 변수로 재면 얼마가 나오나.

선행연구(윤양현 외 2022 등)는 사건기업 1 : 정상기업 1로 맞춘 표본에 무작위 80/20 분할을
쓴다. 그 설정의 정확도 82.73%가 이 분야의 기준점처럼 인용된다.

우리 전수 표본에서는 사건률이 4.2%라 PR-AUC가 0.4 수준이다. 같은 변수를 1:1 설정에 넣으면
**사건률이 50%가 되므로 지표가 전부 올라간다.** 얼마나 올라가는지, 그리고 A·B 격차가
그 설정에서도 남는지 본다.

    반복       표본 추출과 분할을 200회
    분할       무작위 80/20 (논문 설정) · 시간순도 함께
    조건       A · B · A+ · B+

**이 표는 성능 자랑이 아니라 "설정을 바꾸면 숫자가 어떻게 움직이는지"를 보여주는 용도다.**

출력: results/final20.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

import final8 as f8

warnings.filterwarnings("ignore")
RES = Path(__file__).resolve().parent.parent / "results"
N_REP = 200


def fit_eval(X, y, tr, te, seed):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15,
                             min_child_samples=10, class_weight="balanced",
                             verbose=-1, random_state=seed)
    lo, hi = X[tr].quantile(.01), X[tr].quantile(.99)
    clf.fit(X[tr].clip(lo, hi, axis=1), y[tr])
    p = clf.predict_proba(X[te].clip(lo, hi, axis=1))[:, 1]
    return {"정확도": accuracy_score(y[te], p >= .5),
            "AUC": roc_auc_score(y[te], p),
            "PR-AUC": average_precision_score(y[te], p)}


def main():
    rng = np.random.default_rng(42)
    A, B, M, CB, CR = f8.kosdaq_design()
    sel = [c for c in f8.selected() if c in CB.columns]
    conds = {"A": A, "B": B,
             "A+": pd.concat([A, CB[sel], CR], axis=1),
             "B+": pd.concat([B, CB[sel], CR], axis=1)}
    y_all = M.y.values
    ev_idx = np.where(y_all == 1)[0]
    no_idx = np.where(y_all == 0)[0]
    print(f"전체 {len(M):,}행 / 사건 {len(ev_idx)} (사건률 {y_all.mean():.1%})", flush=True)

    rows = []
    for split in ("무작위 80/20", "시간순"):
        acc = {k: [] for k in conds}
        for rep in range(N_REP):
            pick = np.concatenate([ev_idx, rng.choice(no_idx, len(ev_idx), replace=False)])
            rng.shuffle(pick)
            yy = y_all[pick]
            if split == "무작위 80/20":
                cut = int(len(pick) * .8)
                tr_m = np.zeros(len(pick), bool); tr_m[:cut] = True
            else:
                yr = M.pred_year.values[pick]
                tr_m = yr <= 2022
            te_m = ~tr_m
            if yy[te_m].sum() < 5 or yy[tr_m].sum() < 5:
                continue
            for name, X in conds.items():
                s = fit_eval(X.iloc[pick].reset_index(drop=True), yy, tr_m, te_m, rep)
                acc[name].append(s)
            if (rep + 1) % 50 == 0:
                print(f"  [{split}] {rep+1}/{N_REP}", flush=True)
        for name, lst in acc.items():
            d = pd.DataFrame(lst)
            rows.append({"분할": split, "조건": name, "반복": len(d),
                         **{c: round(float(d[c].mean()), 4) for c in d.columns},
                         **{f"{c}_표준편차": round(float(d[c].std()), 4) for c in d.columns}})
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "final20.csv", index=False, encoding="utf-8-sig")
    print("\n" + out[["분할", "조건", "정확도", "AUC", "PR-AUC"]].to_string(index=False))


if __name__ == "__main__":
    main()
