import io
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import requests
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

CODA_API_TOKEN = "4840c197-ef3c-424a-9c1c-7cb580cd7b06"
CODA_DOC_ID = "8efB__Jjpq"
CODA_TABLE_NAME = "PMS PMCF Analysis"

st.set_page_config(page_title="AI PMS/PMCF Intelligence Platform", layout="wide")


def send_analysis_to_coda(run_id, complaint_text, pmcf_text, signal, risk, summary, related_hazard):
    encoded_table_name = quote(CODA_TABLE_NAME, safe="")
    url = f"https://coda.io/apis/v1/docs/{CODA_DOC_ID}/tables/{encoded_table_name}/rows"

    headers = {
        "Authorization": f"Bearer {CODA_API_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "rows": [
            {
                "cells": [
                    {"column": "Run ID", "value": run_id},
                    {"column": "Complaint Text", "value": complaint_text},
                    {"column": "PMCF Text", "value": pmcf_text},
                    {"column": "Signal", "value": str(signal)},
                    {"column": "Risk", "value": str(risk)},
                    {"column": "Summary", "value": str(summary)},
                    {"column": "Related Hazard", "value": related_hazard},
                    {"column": "Created At", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.status_code, response.text


ISSUE_KEYWORDS: Dict[str, List[str]] = {
    "Breakage": ["break", "broken", "fracture", "snap", "파손", "부러", "깨짐"],
    "Migration/Slip": ["slip", "migrat", "move", "displace", "이탈", "미끄", "이동"],
    "Inflammation/Irritation": ["inflamm", "redness", "swelling", "irrit", "염증", "발적", "붓", "자극"],
    "Pain/Discomfort": ["pain", "discomfort", "ache", "통증", "불편", "아픔"],
    "No Issue": ["no issue", "none", "stable", "정상", "문제 없음", "이상 없음", "없음"],
    "Other": []
}

NEGATIVE_WORDS = [
    "break", "slip", "migrat", "inflamm", "pain", "discomfort", "adverse",
    "파손", "염증", "통증", "불편", "이탈"
]
POSITIVE_WORDS = [
    "no issue", "stable", "resolved", "good", "favorable",
    "문제 없음", "안정", "호전", "양호"
]

PMCF_REQUIRED_COLUMNS = [
    "Response_ID",
    "Patient_ID",
    "Country",
    "Followup_Period",
    "SAE",
    "Hemorrhage",
    "Infection",
    "Migration",
    "Performance_Rating",
]

FOLLOWUP_ORDER = ["<6m", "6-12m", "1-3y", ">3y"]
PERFORMANCE_LABELS = {
    5: "Excellent",
    4: "Good",
    3: "Satisfactory",
    2: "Poor",
    1: "Unsure",
}


def normalize_lines(text: str) -> List[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-•").strip()
        if line:
            lines.append(line)
    return lines


def classify_issue(text: str) -> str:
    t = str(text).lower()
    for category, kws in ISSUE_KEYWORDS.items():
        if category == "Other":
            continue
        for kw in kws:
            if kw in t:
                return category
    return "Other"


def pmcf_sentiment(text: str) -> str:
    t = str(text).lower()
    neg = any(w in t for w in NEGATIVE_WORDS)
    pos = any(w in t for w in POSITIVE_WORDS)
    if neg and not pos:
        return "Negative"
    if pos and not neg:
        return "Positive"
    return "Neutral"


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    adj = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - adj), min(1.0, center + adj)


def parse_text_inputs(complaint_text: str, pmcf_text: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comp = pd.DataFrame({"text": normalize_lines(complaint_text)})
    if not comp.empty:
        comp["source"] = "Complaint"
        comp["issue_category"] = comp["text"].apply(classify_issue)

    pmcf = pd.DataFrame({"text": normalize_lines(pmcf_text)})
    if not pmcf.empty:
        pmcf["source"] = "PMCF"
        pmcf["sentiment"] = pmcf["text"].apply(pmcf_sentiment)
    return comp, pmcf


def normalize_binary(series: pd.Series) -> pd.Series:
    mapping = {
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "true": 1,
        "false": 0,
    }

    def _convert(x):
        if pd.isna(x):
            return pd.NA
        if isinstance(x, (int, float)) and x in [0, 1]:
            return int(x)
        s = str(x).strip().lower()
        return mapping.get(s, pd.NA)

    return series.apply(_convert)


def parse_uploaded_complaint(file) -> pd.DataFrame:
    filename = file.name.lower()

    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="cp949")
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(file, sheet_name=0)
    else:
        raise ValueError("Complaint 파일은 CSV/XLSX/XLS 형식만 지원합니다.")

    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    text_col = None
    for c in ["text", "description", "complaint", "comment", "issue"]:
        if c in lower_map:
            text_col = lower_map[c]
            break

    if text_col is None:
        raise ValueError("Complaint 파일에 text / description / complaint / comment / issue 컬럼 중 하나가 필요합니다.")

    out = df.rename(columns={text_col: "text"}).copy()
    out["text"] = out["text"].astype(str).str.strip()

    if "date" in lower_map:
        out["date"] = pd.to_datetime(out[lower_map["date"]], errors="coerce")

    return out


def validate_pmcf_columns(df: pd.DataFrame) -> None:
    missing = [col for col in PMCF_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"PMCF 필수 컬럼 누락: {', '.join(missing)}")


def preprocess_pmcf_excel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

    for col in ["SAE", "Hemorrhage", "Infection", "Migration"]:
        if col in df.columns:
            df[col] = normalize_binary(df[col])

    if "Performance_Rating" in df.columns:
        df["Performance_Rating"] = pd.to_numeric(df["Performance_Rating"], errors="coerce")

    for col in ["Surgery_Date", "Followup_Date", "Completion_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Followup_Period" in df.columns:
        df["Followup_Period"] = df["Followup_Period"].astype("string").str.strip()
        df["Followup_Period"] = pd.Categorical(
            df["Followup_Period"],
            categories=FOLLOWUP_ORDER,
            ordered=True,
        )

    return df


def pmcf_excel_to_text_records(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_pmcf_excel(df)

    def build_text(row: pd.Series) -> str:
        parts = []

        if row.get("SAE") == 1:
            parts.append("serious adverse event observed")
        if row.get("Hemorrhage") == 1:
            parts.append("clinically significant hemorrhage observed")
        if row.get("Infection") == 1:
            parts.append("postoperative infection or inflammation observed")
        if row.get("Migration") == 1:
            parts.append("clip migration or displacement observed")

        lig = str(row.get("Ligation_Result", "")).strip().lower()
        if lig == "complete":
            parts.append("complete ligation achieved")
        elif lig == "partial":
            parts.append("partial ligation and re-application needed")
        elif lig == "fail":
            parts.append("ligation failed")

        performance = row.get("Performance_Rating")
        if pd.notna(performance):
            perf_map = {
                5: "overall performance excellent",
                4: "overall performance good",
                3: "overall performance satisfactory",
                2: "overall performance poor",
                1: "overall performance unsure",
            }
            parts.append(perf_map.get(int(performance), "performance not specified"))

        if not parts:
            parts.append("no issue observed")

        return "; ".join(parts)

    out = df.copy()
    out["text"] = out.apply(build_text, axis=1)
    out["source"] = "PMCF"
    out["sentiment"] = out["text"].apply(pmcf_sentiment)
    return out


def parse_uploaded_pmcf(file) -> pd.DataFrame:
    filename = file.name.lower()

    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="cp949")

        df.columns = [str(c).strip() for c in df.columns]
        lower_map = {c.lower(): c for c in df.columns}

        text_col = None
        for c in ["text", "description", "pmcf", "comment", "assessment", "issue"]:
            if c in lower_map:
                text_col = lower_map[c]
                break

        if text_col is not None:
            out = df.rename(columns={text_col: "text"}).copy()
            out["text"] = out["text"].astype(str).str.strip()
            if "date" in lower_map:
                out["date"] = pd.to_datetime(out[lower_map["date"]], errors="coerce")
            out["source"] = "PMCF"
            out["sentiment"] = out["text"].apply(pmcf_sentiment)
            return out

        validate_pmcf_columns(df)
        return pmcf_excel_to_text_records(df)

    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        xls = pd.ExcelFile(file)
        if "Raw_Data" in xls.sheet_names:
            df = pd.read_excel(file, sheet_name="Raw_Data")
        else:
            df = pd.read_excel(file, sheet_name=0)

        validate_pmcf_columns(df)
        return pmcf_excel_to_text_records(df)

    else:
        raise ValueError("PMCF 파일은 CSV/XLSX/XLS 형식만 지원합니다.")


def maybe_merge(comp: pd.DataFrame, pmcf: pd.DataFrame, comp_file, pmcf_file) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if comp_file is not None:
        uploaded = parse_uploaded_complaint(comp_file)
        uploaded["source"] = "Complaint"
        uploaded["issue_category"] = uploaded["text"].apply(classify_issue)
        comp = pd.concat([comp, uploaded], ignore_index=True) if not comp.empty else uploaded

    if pmcf_file is not None:
        uploaded = parse_uploaded_pmcf(pmcf_file)
        uploaded["source"] = "PMCF"
        if "sentiment" not in uploaded.columns:
            uploaded["sentiment"] = uploaded["text"].apply(pmcf_sentiment)
        pmcf = pd.concat([pmcf, uploaded], ignore_index=True) if not pmcf.empty else uploaded

    return comp, pmcf


def make_issue_table(comp: pd.DataFrame) -> pd.DataFrame:
    total = len(comp)
    rows = []
    for cat, count in comp["issue_category"].value_counts().items():
        lo, hi = wilson_ci(int(count), total)
        rows.append({
            "Issue": cat,
            "Count": int(count),
            "Rate (%)": round(count / total * 100, 1) if total else 0.0,
            "95% CI Low (%)": round(lo * 100, 1),
            "95% CI High (%)": round(hi * 100, 1),
        })
    return pd.DataFrame(rows)


def make_pmcf_table(pmcf: pd.DataFrame) -> pd.DataFrame:
    total = len(pmcf)
    rows = []
    for cat, count in pmcf["sentiment"].value_counts().items():
        lo, hi = wilson_ci(int(count), total)
        rows.append({
            "PMCF Assessment": cat,
            "Count": int(count),
            "Rate (%)": round(count / total * 100, 1) if total else 0.0,
            "95% CI Low (%)": round(lo * 100, 1),
            "95% CI High (%)": round(hi * 100, 1),
        })
    return pd.DataFrame(rows)


def trend_by_month(df: pd.DataFrame, category_col: str) -> Optional[pd.DataFrame]:
    if "date" not in df.columns or df["date"].isna().all():
        return None
    tmp = df.dropna(subset=["date"]).copy()
    if tmp.empty:
        return None
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
    return tmp.groupby(["month", category_col]).size().reset_index(name="count")


def detect_signal(comp: pd.DataFrame, pmcf: pd.DataFrame) -> Tuple[str, str]:
    total_c = len(comp)
    total_p = len(pmcf)

    breakage = (comp["issue_category"] == "Breakage").sum() if total_c else 0
    migration = (comp["issue_category"] == "Migration/Slip").sum() if total_c else 0
    negative_pmcf = (pmcf["sentiment"] == "Negative").sum() if total_p else 0

    signal = "None"
    risk = "Low"

    if total_c and (breakage / total_c >= 0.08 or migration / total_c >= 0.08):
        signal = "Potential complaint signal"
        risk = "Moderate"
    if total_p and negative_pmcf / total_p >= 0.20:
        signal = "Potential PMCF signal"
        risk = "Moderate"
    if total_c and total_p and ((breakage + migration) / total_c >= 0.12 and negative_pmcf / total_p >= 0.20):
        signal = "Combined safety signal"
        risk = "High"

    return signal, risk


def fallback_summary(issue_table: pd.DataFrame, pmcf_table: pd.DataFrame, signal: str, risk: str) -> str:
    top_issue = issue_table.iloc[0]["Issue"] if not issue_table.empty else "No issue identified"
    top_rate = issue_table.iloc[0]["Rate (%)"] if not issue_table.empty else 0
    negative_rate = 0
    if not pmcf_table.empty and "Negative" in pmcf_table["PMCF Assessment"].values:
        negative_rate = pmcf_table.loc[pmcf_table["PMCF Assessment"] == "Negative", "Rate (%)"].iloc[0]

    return (
        f"Complaint 분석 기준 최다 이슈는 {top_issue}이며 발생률은 {top_rate}%입니다. "
        f"PMCF 부정 응답 비율은 {negative_rate}%입니다. "
        f"현재 신호 판정은 '{signal}', 전반적 위험 수준은 '{risk}'입니다. "
        "현 단계에서는 담당자의 규제/임상 검토를 전제로 PSUR 초안 작성에 활용할 수 있습니다."
    )


def ask_gpt(api_key: str, model: str, issue_table: pd.DataFrame, pmcf_table: pd.DataFrame, signal: str, risk: str) -> Optional[str]:
    if not api_key or OpenAI is None:
        return None

    client = OpenAI(api_key=api_key)
    prompt = f'''
You are a senior EU MDR PMS/PMCF specialist.
Write a concise regulatory-style summary in Korean.
Use the following analysis results.

Complaint table:
{issue_table.to_dict(orient="records")}

PMCF table:
{pmcf_table.to_dict(orient="records")}

Signal: {signal}
Risk: {risk}

Requirements:
- 5~7 sentences
- professional tone
- mention whether any new safety signal is identified
- mention benefit-risk profile
- mention whether additional review is recommended
'''
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert regulatory writer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def register_korean_font() -> str:
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
        return "HYSMyeongJo-Medium"
    except Exception:
        pass

    candidates = [
        r"C:\\Windows\\Fonts\\malgun.ttf",
        r"C:\\Windows\\Fonts\\gulim.ttc",
        r"C:\\Windows\\Fonts\\batang.ttc",
    ]
    for path in candidates:
        try:
            pdfmetrics.registerFont(TTFont("KoreanFont", path))
            return "KoreanFont"
        except Exception:
            continue

    return "Helvetica"


def wrap_text(text: str, max_chars: int) -> List[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = f"{current} {word}".strip()
        if len(test) <= max_chars:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def generate_pdf(issue_table: pd.DataFrame, pmcf_table: pd.DataFrame, summary: str, signal: str, risk: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    font_name = register_korean_font()
    _, height = A4

    y = height - 40
    c.setFont(font_name, 16)
    c.drawString(40, y, "AI PMS/PMCF 분석 보고서")
    y -= 28

    c.setFont(font_name, 10)
    c.drawString(40, y, f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 24
    c.drawString(40, y, f"Signal Detection: {signal}")
    y -= 18
    c.drawString(40, y, f"Risk Level: {risk}")
    y -= 26

    c.setFont(font_name, 12)
    c.drawString(40, y, "1. Executive Summary")
    y -= 18

    c.setFont(font_name, 10)
    for line in wrap_text(summary, 78):
        c.drawString(40, y, line)
        y -= 14
        if y < 60:
            c.showPage()
            c.setFont(font_name, 10)
            y = height - 40

    y -= 8
    c.setFont(font_name, 12)
    c.drawString(40, y, "2. Complaint Issue Table")
    y -= 18

    c.setFont(font_name, 9)
    for _, row in issue_table.iterrows():
        txt = f"{row['Issue']}: Count={row['Count']}, Rate={row['Rate (%)']}%, 95% CI={row['95% CI Low (%)']}~{row['95% CI High (%)']}%"
        for line in wrap_text(txt, 90):
            c.drawString(40, y, line)
            y -= 12
            if y < 60:
                c.showPage()
                c.setFont(font_name, 9)
                y = height - 40

    y -= 8
    c.setFont(font_name, 12)
    c.drawString(40, y, "3. PMCF Assessment Table")
    y -= 18

    c.setFont(font_name, 9)
    for _, row in pmcf_table.iterrows():
        txt = f"{row['PMCF Assessment']}: Count={row['Count']}, Rate={row['Rate (%)']}%, 95% CI={row['95% CI Low (%)']}~{row['95% CI High (%)']}%"
        for line in wrap_text(txt, 90):
            c.drawString(40, y, line)
            y -= 12
            if y < 60:
                c.showPage()
                c.setFont(font_name, 9)
                y = height - 40

    c.save()
    return buf.getvalue()


def plot_issue_bar(issue_table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(issue_table["Issue"], issue_table["Count"])
    ax.set_title("Complaint Issue Frequency")
    ax.set_ylabel("Count")
    ax.set_xlabel("Issue")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


def plot_pmcf_bar(pmcf_table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(pmcf_table["PMCF Assessment"], pmcf_table["Count"])
    ax.set_title("PMCF Response Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Assessment")
    plt.tight_layout()
    return fig


def plot_monthly(trend_df: pd.DataFrame, label_col: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = trend_df.pivot(index="month", columns=label_col, values="count").fillna(0)
    pivot.plot(ax=ax, marker="o")
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel("Month")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


def safe_count_binary(df: pd.DataFrame, col: str) -> Tuple[int, int, float]:
    if col not in df.columns:
        return 0, 0, 0.0
    valid = df[col].dropna()
    yes_count = int((valid == 1).sum())
    denom = int(valid.shape[0])
    return yes_count, denom, round((yes_count / denom) * 100, 1) if denom else 0.0


def build_structured_pmcf_kpis(pmcf: pd.DataFrame) -> Optional[dict]:
    if not set(PMCF_REQUIRED_COLUMNS).issubset(set(pmcf.columns)):
        return None

    sae_yes, _, sae_rate = safe_count_binary(pmcf, "SAE")
    he_yes, _, he_rate = safe_count_binary(pmcf, "Hemorrhage")
    inf_yes, _, inf_rate = safe_count_binary(pmcf, "Infection")
    mig_yes, _, mig_rate = safe_count_binary(pmcf, "Migration")

    complete = 0
    complete_rate = 0.0
    if "Ligation_Result" in pmcf.columns:
        lig = pmcf["Ligation_Result"].dropna().astype(str).str.strip().str.lower()
        denom = int(lig.shape[0])
        complete = int((lig == "complete").sum())
        complete_rate = round((complete / denom) * 100, 1) if denom else 0.0

    return {
        "responses": int(pmcf.shape[0]),
        "sae_yes": sae_yes,
        "hemorrhage_yes": he_yes,
        "infection_yes": inf_yes,
        "migration_yes": mig_yes,
        "sae_rate": sae_rate,
        "hemorrhage_rate": he_rate,
        "infection_rate": inf_rate,
        "migration_rate": mig_rate,
        "complete_ligation": complete,
        "complete_ligation_rate": complete_rate,
    }


def performance_distribution(pmcf: pd.DataFrame) -> Optional[pd.Series]:
    if "Performance_Rating" not in pmcf.columns:
        return None
    return (
        pmcf["Performance_Rating"]
        .dropna()
        .astype(int)
        .map(PERFORMANCE_LABELS)
        .value_counts()
        .reindex(["Excellent", "Good", "Satisfactory", "Poor", "Unsure"], fill_value=0)
    )


st.title("🤖 AI PMS / PMCF Intelligence Platform")
st.caption("Complaint/PMCF 분석, 실제 통계 계산, 그래프 생성, GPT 요약, PDF 보고서 출력")

with st.sidebar:
    st.header("설정")
    api_key = st.text_input("OpenAI API Key (선택)", type="password")
    model = st.selectbox("GPT Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
    st.markdown("---")
    st.write("Complaint 업로드 권장 컬럼")
    st.code("text, date", language=None)
    st.write("PMCF 업로드 지원 형식")
    st.code("CSV(text/date) 또는 XLSX Raw_Data", language=None)
    st.caption("PMCF Excel은 Raw_Data 시트를 우선 인식합니다.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Automation", "Complaint 분류")
col2.metric("Advanced Output", "통계 + 그래프")
col3.metric("Generative AI", "GPT 요약")
col4.metric("Report", "PDF 다운로드")

st.divider()

c1, c2 = st.columns(2)
with c1:
    complaint_text = st.text_area(
        "Complaint Data (한 줄당 1건)",
        height=180,
        placeholder="예:\nClip breakage observed during procedure\nClip slipped during surgery\nNo issue observed",
    )
    complaint_file = st.file_uploader("Complaint 파일 업로드", type=["csv", "xlsx", "xls"], key="comp")

with c2:
    pmcf_text = st.text_area(
        "PMCF Data (한 줄당 1건)",
        height=180,
        placeholder="예:\nNo complication observed after 6 months\nInitial mild discomfort resolved without intervention",
    )
    pmcf_file = st.file_uploader("PMCF 파일 업로드", type=["csv", "xlsx", "xls"], key="pmcf")

related_hazard = st.text_input("Related Hazard ID (예: HZ-001)")

if st.button("🚀 Run Full Analysis", use_container_width=True):
    with st.spinner("데이터 정리 및 분석 중..."):
        try:
            complaint_df, pmcf_df = parse_text_inputs(complaint_text, pmcf_text)
            complaint_df, pmcf_df = maybe_merge(complaint_df, pmcf_df, complaint_file, pmcf_file)
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
            st.stop()

        if complaint_df.empty and pmcf_df.empty:
            st.error("Complaint 또는 PMCF 데이터를 입력하거나 파일을 업로드해 주세요.")
            st.stop()

        issue_table = (
            make_issue_table(complaint_df)
            if not complaint_df.empty
            else pd.DataFrame(columns=["Issue", "Count", "Rate (%)", "95% CI Low (%)", "95% CI High (%)"])
        )

        pmcf_table = (
            make_pmcf_table(pmcf_df)
            if not pmcf_df.empty
            else pd.DataFrame(columns=["PMCF Assessment", "Count", "Rate (%)", "95% CI Low (%)", "95% CI High (%)"])
        )

        signal, risk = detect_signal(complaint_df, pmcf_df)

        gpt_summary = None
        if not issue_table.empty or not pmcf_table.empty:
            try:
                gpt_summary = ask_gpt(api_key, model, issue_table, pmcf_table, signal, risk)
            except Exception as e:
                st.warning(f"GPT 요약 생성에 실패했습니다. 로컬 요약으로 대체합니다. ({e})")

        summary = gpt_summary or fallback_summary(issue_table, pmcf_table, signal, risk)
        pdf_bytes = generate_pdf(issue_table, pmcf_table, summary, signal, risk)
        pmcf_kpis = build_structured_pmcf_kpis(pmcf_df)

        run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        status_code, response_text = send_analysis_to_coda(
            run_id=run_id,
            complaint_text=complaint_text,
            pmcf_text=pmcf_text,
            signal=signal,
            risk=risk,
            summary=summary,
            related_hazard=related_hazard
        )

        if status_code == 202:
            st.success("분석 완료 + Coda 저장 성공")
        else:
            st.warning(f"Coda 저장 실패: {status_code}")
            st.text(response_text)

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Complaint Count", len(complaint_df))
    kc2.metric("PMCF Count", len(pmcf_df))
    kc3.metric("Signal Detection", signal)
    kc4.metric("Risk Level", risk)

    st.subheader("1) 실제 통계 결과")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("**Complaint Issue Table**")
        st.dataframe(issue_table, use_container_width=True)
    with tc2:
        st.markdown("**PMCF Assessment Table**")
        st.dataframe(pmcf_table, use_container_width=True)

    if pmcf_kpis is not None:
        st.subheader("2) Structured PMCF KPI")
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        pc1.metric("SAE Rate", f"{pmcf_kpis['sae_rate']}%")
        pc2.metric("Hemorrhage Rate", f"{pmcf_kpis['hemorrhage_rate']}%")
        pc3.metric("Infection Rate", f"{pmcf_kpis['infection_rate']}%")
        pc4.metric("Migration Rate", f"{pmcf_kpis['migration_rate']}%")
        pc5.metric("1st Ligation Success", f"{pmcf_kpis['complete_ligation_rate']}%")

        perf = performance_distribution(pmcf_df)
        if perf is not None:
            st.markdown("**Performance Distribution**")
            st.bar_chart(perf)

        bd1, bd2 = st.columns(2)
        with bd1:
            if "Followup_Period" in pmcf_df.columns:
                followup_dist = (
                    pmcf_df["Followup_Period"]
                    .value_counts(dropna=False)
                    .reindex(FOLLOWUP_ORDER, fill_value=0)
                )
                st.markdown("**Follow-up Period Distribution**")
                st.bar_chart(followup_dist)

        with bd2:
            if "Country" in pmcf_df.columns:
                country_dist = pmcf_df["Country"].dropna().astype(str).value_counts().reset_index()
                country_dist.columns = ["Country", "Count"]
                st.markdown("**Country Distribution**")
                st.bar_chart(country_dist.set_index("Country"))

    st.subheader("3) 그래프")
    gc1, gc2 = st.columns(2)
    with gc1:
        if not issue_table.empty:
            st.pyplot(plot_issue_bar(issue_table))
        else:
            st.info("Complaint 데이터가 없어 그래프를 생성하지 않았습니다.")
    with gc2:
        if not pmcf_table.empty:
            st.pyplot(plot_pmcf_bar(pmcf_table))
        else:
            st.info("PMCF 데이터가 없어 그래프를 생성하지 않았습니다.")

    comp_trend = trend_by_month(complaint_df, "issue_category") if not complaint_df.empty else None
    pmcf_trend = trend_by_month(pmcf_df, "sentiment") if not pmcf_df.empty else None

    if comp_trend is not None or pmcf_trend is not None:
        st.subheader("4) 월별 트렌드")
        if comp_trend is not None:
            st.pyplot(plot_monthly(comp_trend, "issue_category", "Complaint Monthly Trend"))
        if pmcf_trend is not None:
            st.pyplot(plot_monthly(pmcf_trend, "sentiment", "PMCF Monthly Trend"))

    st.subheader("5) GPT 요약 / PSUR Summary")
    st.write(summary)

    st.download_button(
        "📄 PDF 보고서 다운로드",
        data=pdf_bytes,
        file_name=f"AI_PMS_PMCF_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
    )

    with st.expander("원본 정리 데이터 보기"):
        if not complaint_df.empty:
            st.markdown("**Complaint Records**")
            st.dataframe(complaint_df, use_container_width=True)
        if not pmcf_df.empty:
            st.markdown("**PMCF Records**")
            st.dataframe(pmcf_df, use_container_width=True)

st.divider()
st.subheader("Coda 연결 디버그")

st.code(f"CODA_DOC_ID = {CODA_DOC_ID}")
st.code(f"CODA_TABLE_NAME = {CODA_TABLE_NAME}")
st.code(f"TOKEN PREFIX = {CODA_API_TOKEN[:8]}...")

if st.button("🔍 Coda 문서 목록 조회"):
    headers = {"Authorization": f"Bearer {CODA_API_TOKEN}"}
    res = requests.get("https://coda.io/apis/v1/docs", headers=headers)
    st.write("Status:", res.status_code)
    try:
        st.json(res.json())
    except Exception:
        st.text(res.text)

if st.button("🔎 현재 DOC_ID 직접 조회"):
    headers = {"Authorization": f"Bearer {CODA_API_TOKEN}"}
    url = f"https://coda.io/apis/v1/docs/{CODA_DOC_ID}"
    res = requests.get(url, headers=headers)
    st.write("Status:", res.status_code)
    try:
        st.json(res.json())
    except Exception:
        st.text(res.text)     
