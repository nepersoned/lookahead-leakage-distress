"""KIND 공시의 빈 종목코드를 회사명으로 채운다.

KIND 관리종목 공시 자료에서 외국 기업(중국·미국계 지주회사 등, 종목코드 9xxxxx)은 stock_code가
비어 있다. 종목코드로만 DART 고유번호에 연결하면 이 기업들의 지정·해제·상장폐지가 통째로 빠져,
채점 표본에서 '지정 없음'으로 잘못 남는다(2026-11-05 확인: 딥커머스·코오롱티슈진·파이온엑스·
크리스탈신소재). 빈 종목코드를 DART 고유번호 목록의 회사명(종목코드가 있는 회사 중 이름이
하나뿐인 경우)으로 채운다.
"""
from __future__ import annotations

import pandas as pd

_NAME2CODE: dict | None = None


def _name_map() -> dict:
    global _NAME2CODE
    if _NAME2CODE is None:
        from dart import Dart
        c = pd.DataFrame(Dart().corp_codes())
        c["stock_code"] = c.stock_code.astype(str).str.strip()
        c = c[c.stock_code.str.fullmatch(r"\d{6}")]
        c["corp_name"] = c.corp_name.astype(str).str.strip()
        uniq = c.groupby("corp_name").stock_code.nunique()
        c = c[c.corp_name.isin(uniq[uniq == 1].index)]
        _NAME2CODE = c.drop_duplicates("corp_name").set_index("corp_name")["stock_code"].to_dict()
    return _NAME2CODE


def fill(kind: pd.DataFrame, col: str = "stock_code") -> pd.DataFrame:
    """stock_code를 6자리 문자열로 정리하고, 비어 있으면 corp_name으로 채운다."""
    kind = kind.copy()
    sc = kind[col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    sc = sc.where(sc.str.fullmatch(r"\d{1,6}"), None).str.zfill(6)
    miss = sc.isna()
    if miss.any():
        sc[miss] = kind.loc[miss, "corp_name"].astype(str).str.strip().map(_name_map())
    kind[col] = sc
    return kind
