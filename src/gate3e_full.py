"""본 수집 데이터로 실험 A/B를 다시 잰다 (확대 표본).

파일럿(416개사)이 아니라 본 수집(1,737개사)의 재무비율을 쓰고,
라벨은 KIND 확정 지정을 쓴다. 나머지 설계는 gate3c와 동일하다 —
정렬 기준 하나만 바꾸고 표본·변수·모델·분할·지표는 전부 같게 둔다.

    실험 A : 회계연도 정렬. FY(Y-1)을 T에 이미 안다고 가정 (관행)
    실험 B : 접수일 정렬. T 이전에 실제로 공개된 최신 사업보고서만 사용

사용:
    python src/gate3e_full.py --month 2       # 한 시점
    python src/gate3e_full.py --sweep         # 1~7월
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"
OUT = ROOT / "results"

RATIO_COLS = ["debt_ratio", "equity_ratio", "current_ratio", "roa", "op_margin",
              "asset_turnover", "retained_to_ta", "op_to_liab", "impair_ratio",
              "net_margin"]

# 재무비율이 FY2015~2023이므로 예측연도는 2017~2024가 된다.
TRAIN_YEARS = [2017, 2018, 2019, 2020, 2021]
TEST_YEARS = [2022, 2023, 2024]


def load():
    ratios = pd.read_csv(FULL / "ratios_full.csv", dtype={"corp_code": str})
    ratios["corp_code"] = ratios["corp_code"].str.zfill(8)

    reports = pd.read_csv(FULL / "reports_full.csv", dtype={"corp_code": str},
                          parse_dates=["rcept_dt"])
    reports["corp_code"] = reports["corp_code"].str.zfill(8)
    # 사업보고서만, 원 제출일 우선(정정만 있으면 정정일)
    ann = reports[reports["report_type"] == "사업"]
    orig = ann[~ann["is_amended"]]
    pub = (pd.concat([orig, ann]).sort_values("rcept_dt")
           .groupby(["corp_code", "bsns_year"], as_index=False)["rcept_dt"].first()
           .rename(columns={"rcept_dt": "pub_dt"}))

    # 라벨 — KIND 확정 지정, 기업×연도 최초 (사유변경·상장폐지 공시는 지정이 아니다)
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from dart import Dart

    kind = pd.read_csv(FULL / "kind_events.csv", dtype={"stock_code": str},
                       parse_dates=["rcept_dt"])
    kind = kind[kind["event"] == "지정"].copy()
    import kind_codes
    kind = kind_codes.fill(kind)          # 외국 기업의 빈 종목코드를 회사명으로 채운다
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes["stock_code"].astype(str).str.strip().str.zfill(6)
    m = (codes[codes["stock_code"].str.len() == 6]
         .drop_duplicates("stock_code").set_index("stock_code")["corp_code"])
    kind["corp_code"] = kind["stock_code"].map(m)
    kind = kind.dropna(subset=["corp_code"])
    kind["corp_code"] = kind["corp_code"].astype(str).str.zfill(8)
    labels = (kind.sort_values("rcept_dt")
              .groupby(["corp_code", "year"], as_index=False).first()
              .rename(columns={"rcept_dt": "event_dt"})[["corp_code", "event_dt"]])
    return ratios, pub, labels


MARKET = "KOSDAQ"          # in_status가 볼 시장. 코스피 설계를 만들 때는 "KOSPI"로 바꾼다
_KIND_FILES = {"KOSDAQ": ("kind_events_kosdaq_old.csv", "kind_events.csv"),
               "KOSPI": ("kind_events_kospi_old.csv", "kind_events_kospi.csv")}
_STATUS: dict = {}


def status_history(market: str | None = None) -> dict:
    """기업별 관리종목 상태 변화 [(일시, 'on'|'off')]. 지정 → on, '관리종목 해제'(전부 해제)·상장폐지 → off.

    '사유 일부 해제'는 관리종목 상태가 이어지므로 off가 아니다.
    """
    market = market or MARKET
    if market in _STATUS:
        return _STATUS[market]
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from dart import Dart
    kd = pd.concat([pd.read_csv(FULL / f, dtype={"stock_code": str}) for f in _KIND_FILES[market]
                    if (FULL / f).exists()], ignore_index=True)
    kd["dt"] = pd.to_datetime(kd["rcept_dt"].astype(str).str[:19], errors="coerce")
    import kind_codes
    kd = kind_codes.fill(kd)              # 외국 기업의 빈 종목코드를 회사명으로 채운다
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes["stock_code"].astype(str).str.strip().str.zfill(6)
    m = (codes[codes["stock_code"].str.len() == 6]
         .drop_duplicates("stock_code").set_index("stock_code")["corp_code"])
    kd["corp_code"] = kd["stock_code"].map(m)
    kd = kd.dropna(subset=["corp_code", "dt"])
    kd["corp_code"] = kd["corp_code"].astype(str).str.zfill(8)
    full_release = kd["report_nm"].astype(str).str.replace(" ", "").str.startswith("관리종목해제")
    kd["kind"] = ""
    kd.loc[kd["event"] == "지정", "kind"] = "on"
    kd.loc[full_release | (kd["event"] == "상장폐지"), "kind"] = "off"
    kd = kd[kd["kind"] != ""].sort_values("dt")
    _STATUS[market] = {c: list(zip(g.dt, g.kind)) for c, g in kd.groupby("corp_code")}
    return _STATUS[market]


def in_status(code: str, T: pd.Timestamp) -> bool:
    """T 직전에 관리종목 상태였는가 — T 이전 정보만 쓴다."""
    st = False
    for d, k in status_history().get(code, []):
        if d >= T:
            break
        st = k == "on"
    return st


def build(ratios, pub, labels, years, month, horizon=365):
    """예측 시점 T별로 A·B 두 벌의 설계행렬을 만든다. 행과 라벨은 완전 동일."""
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
            if (code, fy_a) not in r.index or (code, fy_b) not in r.index:
                continue
            # T 시점에 관리종목 상태인 기업은 모집단에서 제외한다. 판단에는 T 이전 정보만 쓴다.
            # (2026-10-02 이전 판은 "이전 지정 이력이 있고 올해 재지정되지 않은 기업"을 뺐다 —
            #  라벨을 보고 표본을 고른 것이라 이전 지정 기업이 전부 사건으로 남았다)
            if in_status(code, T):
                continue
            dates = ev.get(code, [])
            hit = any(T <= d < T + pd.Timedelta(days=horizon) for d in dates)
            A.append(r.loc[(code, fy_a), RATIO_COLS].to_dict())
            B.append(r.loc[(code, fy_b), RATIO_COLS].to_dict())
            M.append({"corp_code": code, "pred_year": y, "y": int(hit),
                      "lag": fy_a - fy_b})
    return (pd.DataFrame(A).reset_index(drop=True),
            pd.DataFrame(B).reset_index(drop=True),
            pd.DataFrame(M).reset_index(drop=True))


def evaluate(X, M, tag, seed=42):
    tr = M["pred_year"].isin(TRAIN_YEARS).values
    te = M["pred_year"].isin(TEST_YEARS).values
    lo, hi = X[tr].quantile(0.01), X[tr].quantile(0.99)
    Xtr, Xte = X[tr].clip(lo, hi, axis=1), X[te].clip(lo, hi, axis=1)
    ytr, yte = M.loc[tr, "y"].values, M.loc[te, "y"].values

    out = []
    logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))
    logit.fit(Xtr, ytr)
    p = logit.predict_proba(Xte)[:, 1]
    out.append({"exp": tag, "model": "Logistic", "auc": roc_auc_score(yte, p),
                "ap": average_precision_score(yte, p)})

    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                             min_child_samples=10, class_weight="balanced",
                             verbose=-1, random_state=seed)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    out.append({"exp": tag, "model": "LightGBM", "auc": roc_auc_score(yte, p),
                "ap": average_precision_score(yte, p)})
    return out


def run(month, ratios, pub, labels, verbose=True):
    years = TRAIN_YEARS + TEST_YEARS
    A, B, M = build(ratios, pub, labels, years, month)
    if M.empty or M["y"].sum() == 0:
        raise SystemExit("표본 또는 양성 라벨이 없습니다.")
    te = M["pred_year"].isin(TEST_YEARS)
    if verbose:
        print(f"\n예측 시점 {month}월 1일 | 관측 {len(M):,}행 "
              f"(사건 {int(M.y.sum())}건, {M.y.mean():.2%})")
        print(f"  학습 {int((~te).sum()):,}행 (사건 {int(M.loc[~te,'y'].sum())}) / "
              f"평가 {int(te.sum()):,}행 (사건 {int(M.loc[te,'y'].sum())})")
        print(f"  A와 B의 회계연도 시차: {dict(M.lag.value_counts().sort_index())}")
    res = pd.DataFrame(evaluate(A, M, "A(관행)") + evaluate(B, M, "B(정합)"))
    res["month"] = month
    res["n"] = len(M)
    res["n_event_test"] = int(M.loc[te, "y"].sum())
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--month", type=int, default=2)
    p.add_argument("--sweep", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("자료 로드 중...")
    ratios, pub, labels = load()
    print(f"  재무비율 {len(ratios):,}건 / {ratios.corp_code.nunique():,}개사")
    print(f"  라벨 {len(labels):,}건 / {labels.corp_code.nunique():,}개사")

    months = range(1, 8) if a.sweep else [a.month]
    allres = pd.concat([run(m, ratios, pub, labels, verbose=not a.sweep)
                        for m in months], ignore_index=True)

    name = "full_sweep.csv" if a.sweep else "full_result.csv"
    allres.to_csv(OUT / name, index=False, encoding="utf-8-sig")

    print("\n=== A−B 격차 ===")
    for metric, label in (("ap", "PR-AUC"), ("auc", "AUC")):
        piv = allres.pivot_table(index=["month", "model"], columns="exp",
                                 values=metric)
        piv["격차"] = piv["A(관행)"] - piv["B(정합)"]
        print(f"\n[{label}]")
        print(piv.round(3).to_string())
    print(f"\n→ {OUT / name}")


if __name__ == "__main__":
    main()
