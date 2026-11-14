"""업종 분류표 — 임베딩 결과를 확인할 기준.

DART company.json의 induty_code(표준산업분류)를 기업별로 받는다.
이웃으로 잡힌 기업이 같은 업종인지 보는 데 쓴다.

주의 — 업종 일치율 100%가 목표가 아니다. 텍스트 임베딩이 업종코드를 그대로
복원하면, 비싼 방법으로 업종코드를 재발명한 셈이 된다. Hoberg & Phillips(2016)의
기여는 오히려 텍스트 기반 이웃이 SIC와 **다르다**는 데 있었다.
겹치되 똑같지는 않은 상태가 우리가 원하는 것이다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dart import Dart, DartError

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"

# 표준산업분류 대분류 — induty_code 앞 2자리로 묶는다
SECTION = {
    "01": "농림어업", "05": "광업", "10": "제조-식품", "11": "제조-음료",
    "13": "제조-섬유", "14": "제조-의복", "15": "제조-가죽", "16": "제조-목재",
    "17": "제조-펄프", "18": "제조-인쇄", "19": "제조-코크스", "20": "제조-화학",
    "21": "제조-의약", "22": "제조-고무플라스틱", "23": "제조-비금속",
    "24": "제조-1차금속", "25": "제조-금속가공", "26": "제조-전자부품",
    "27": "제조-의료정밀", "28": "제조-전기장비", "29": "제조-기계",
    "30": "제조-자동차", "31": "제조-운송장비", "32": "제조-가구",
    "33": "제조-기타", "35": "전기가스", "36": "수도", "37": "폐기물",
    "41": "건설", "42": "전문건설", "45": "도매-자동차", "46": "도매",
    "47": "소매", "49": "육상운송", "50": "수상운송", "51": "항공운송",
    "52": "창고운송지원", "55": "숙박", "56": "음식점", "58": "출판",
    "59": "영상음악", "60": "방송", "61": "통신", "62": "컴퓨터프로그래밍",
    "63": "정보서비스", "64": "금융", "65": "보험", "66": "금융지원",
    "68": "부동산", "70": "연구개발", "71": "전문서비스", "72": "건축기술",
    "73": "기타전문과학", "74": "사업지원", "75": "사업지원서비스",
    "85": "교육", "86": "보건", "87": "사회복지", "90": "예술스포츠",
    "91": "스포츠오락", "94": "협회단체", "96": "기타개인서비스",
}


def main() -> None:
    rep = pd.read_csv(FULL / "reports_full.csv", dtype={"corp_code": str})
    rep["corp_code"] = rep["corp_code"].str.zfill(8)
    firms = rep.drop_duplicates("corp_code")[["corp_code", "corp_name",
                                              "stock_code"]]
    print(f"대상 {len(firms):,}개사")

    dart = Dart()
    rows, miss = [], 0
    for i, (_, f) in enumerate(firms.iterrows(), 1):
        try:
            info = dart.get_json("company.json", corp_code=f.corp_code)
        except DartError:
            miss += 1
            continue
        code = (info.get("induty_code") or "").strip()
        rows.append({
            "corp_code": f.corp_code, "corp_name": f.corp_name,
            "stock_code": f.stock_code,
            "induty_code": code,
            "induty_2": code[:2],
            "induty_nm": SECTION.get(code[:2], "미분류"),
            "corp_cls": info.get("corp_cls", ""),
        })
        if i % 300 == 0:
            print(f"  {i}/{len(firms)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(FULL / "industry.csv", index=False, encoding="utf-8-sig")
    print(f"\n업종 {len(df):,}개사 · 조회 실패 {miss}")
    print(f"코드 결측 {int((df.induty_code == '').sum())}건 "
          f"· 미분류 {int((df.induty_nm == '미분류').sum())}건")
    print("\n[대분류 상위 12]")
    print(df.induty_nm.value_counts().head(12).to_string())
    print(f"\n→ {FULL / 'industry.csv'}")


if __name__ == "__main__":
    main()
