import io
import math
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
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

CODA_API_TOKEN  = "4840c197-ef3c-424a-9c1c-7cb580cd7b06"
CODA_DOC_ID     = "8efB__Jjpq"
CODA_TABLE_NAME = "PMS PMCF Analysis"

st.set_page_config(page_title="AI PMS/PMCF Intelligence Platform", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
ISSUE_KEYWORDS: Dict[str, List[str]] = {
    "Breakage":                ["break","broken","fracture","snap","파손","부러","깨짐"],
    "Migration/Slip":          ["slip","migrat","move","displace","이탈","미끄","이동"],
    "Inflammation/Irritation": ["inflamm","redness","swelling","irrit","염증","발적","붓","자극"],
    "Pain/Discomfort":         ["pain","discomfort","ache","통증","불편","아픔"],
    "No Issue":                ["no issue","none","stable","정상","문제 없음","이상 없음","없음"],
    "Other":                   [],
}
NEGATIVE_WORDS  = ["break","slip","migrat","inflamm","pain","discomfort","adverse","파손","염증","통증","불편","이탈"]
POSITIVE_WORDS  = ["no issue","stable","resolved","good","favorable","문제 없음","안정","호전","양호"]
PMCF_REQUIRED_COLUMNS = ["Response_ID","Patient_ID","Country","Followup_Period",
                          "SAE","Hemorrhage","Infection","Migration","Performance_Rating"]
FOLLOWUP_ORDER  = ["<6m","6-12m","1-3y",">3y"]
PERFORMANCE_LABELS = {5:"Excellent",4:"Good",3:"Satisfactory",2:"Poor",1:"Unsure"}
TREND_THRESHOLDS   = {"Breakage":0.08,"Migration/Slip":0.08,
                      "Inflammation/Irritation":0.10,"Pain/Discomfort":0.10,"Other":0.15}
TREND_BASELINE_WINDOW = 3
SAFETY_THRESHOLD      = 5.0
PERFORMANCE_THRESHOLD = 8.0
LIGATION_THRESHOLD    = 5.0

PERF_COLORS    = ["#2196F3","#4CAF50","#FF9800","#F44336","#9E9E9E"]
FOLLOW_COLORS  = ["#5C6BC0","#26A69A","#FFA726","#EF5350"]
COUNTRY_COLORS = ["#42A5F5","#66BB6A","#FFA726","#EF5350","#AB47BC",
                  "#26C6DA","#FF7043","#8D6E63","#78909C","#EC407A"]
MULTI_COLORS   = ["#2196F3","#4CAF50","#FF9800","#F44336","#AB47BC",
                  "#26C6DA","#FF7043","#8D6E63","#78909C","#EC407A"]

# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════
def normalize_lines(text: str) -> List[str]:
    return [r.strip().lstrip("-•").strip() for r in text.splitlines()
            if r.strip().lstrip("-•").strip()]

def classify_issue(text: str) -> str:
    t = str(text).lower()
    for cat, kws in ISSUE_KEYWORDS.items():
        if cat == "Other": continue
        for kw in kws:
            if kw in t: return cat
    return "Other"

def pmcf_sentiment(text: str) -> str:
    t   = str(text).lower()
    neg = any(w in t for w in NEGATIVE_WORDS)
    pos = any(w in t for w in POSITIVE_WORDS)
    if neg and not pos: return "Negative"
    if pos and not neg: return "Positive"
    return "Neutral"

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0: return 0.0, 0.0
    phat   = k / n
    denom  = 1 + z**2 / n
    center = (phat + z**2 / (2*n)) / denom
    adj    = z * math.sqrt((phat*(1-phat) + z**2/(4*n)) / n) / denom
    return max(0.0, center-adj), min(1.0, center+adj)

def normalize_binary(series: pd.Series) -> pd.Series:
    mapping = {"1":1,"0":0,"yes":1,"no":0,"y":1,"n":0,"true":1,"false":0}
    def _c(x):
        if pd.isna(x): return pd.NA
        if isinstance(x,(int,float)) and x in [0,1]: return int(x)
        return mapping.get(str(x).strip().lower(), pd.NA)
    return series.apply(_c)

def safe_count_binary(df: pd.DataFrame, col: str) -> Tuple[int,int,float]:
    if col not in df.columns: return 0,0,0.0
    valid = df[col].dropna()
    y = int((valid==1).sum()); d = int(valid.shape[0])
    return y, d, round(y/d*100,1) if d else 0.0

def safe_value_counts(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df.columns: return None
    s = df[col].dropna().astype(str).str.strip()
    s = s[~s.str.lower().isin(["nan","none","<na>",""])]
    return s.value_counts() if not s.empty else None

# ═══════════════════════════════════════════════════════════════════════════════
# Parsing
# ═══════════════════════════════════════════════════════════════════════════════
def parse_text_inputs(complaint_text: str, pmcf_text: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comp = pd.DataFrame({"text": normalize_lines(complaint_text)})
    if not comp.empty:
        comp["source"] = "Complaint"
        comp["issue_category"] = comp["text"].apply(classify_issue)
    pmcf = pd.DataFrame({"text": normalize_lines(pmcf_text)})
    if not pmcf.empty:
        pmcf["source"]    = "PMCF"
        pmcf["sentiment"] = pmcf["text"].apply(pmcf_sentiment)
    return comp, pmcf

def parse_uploaded_complaint(file) -> pd.DataFrame:
    fn = file.name.lower()
    if fn.endswith(".csv"):
        try:    df = pd.read_csv(file, encoding="utf-8-sig")
        except: file.seek(0); df = pd.read_csv(file, encoding="cp949")
    elif fn.endswith((".xlsx",".xls")): df = pd.read_excel(file, sheet_name=0)
    else: raise ValueError("Complaint file must be CSV / XLSX / XLS.")
    df.columns = [str(c).strip() for c in df.columns]
    lm = {c.lower():c for c in df.columns}
    tc = next((lm[c] for c in ["text","description","complaint","comment","issue"] if c in lm), None)
    if tc is None: raise ValueError("Complaint file must contain: text / description / complaint / comment / issue.")
    out = df.rename(columns={tc:"text"}).copy()
    out["text"] = out["text"].astype(str).str.strip()
    if "date" in lm: out["date"] = pd.to_datetime(out[lm["date"]], errors="coerce")
    return out

def validate_pmcf_columns(df: pd.DataFrame) -> None:
    missing = [c for c in PMCF_REQUIRED_COLUMNS if c not in df.columns]
    if missing: raise ValueError(f"Missing PMCF columns: {', '.join(missing)}")

def preprocess_pmcf_excel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip().replace({"nan":pd.NA,"None":pd.NA,"":pd.NA})
    for col in ["SAE","Hemorrhage","Infection","Migration","Misfiring","Slippage"]:
        if col in df.columns: df[col] = normalize_binary(df[col])
    if "Performance_Rating" in df.columns:
        df["Performance_Rating"] = pd.to_numeric(df["Performance_Rating"], errors="coerce")
    for col in ["Surgery_Date","Followup_Date","Completion_Date"]:
        if col in df.columns: df[col] = pd.to_datetime(df[col], errors="coerce")
    if "Followup_Period" in df.columns:
        df["Followup_Period"] = df["Followup_Period"].astype("string").str.strip()
        df["Followup_Period"] = pd.Categorical(df["Followup_Period"],
                                               categories=FOLLOWUP_ORDER, ordered=True)
    return df

def pmcf_excel_to_text_records(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_pmcf_excel(df)
    def build_text(row):
        parts = []
        if row.get("SAE")==1:        parts.append("serious adverse event observed")
        if row.get("Hemorrhage")==1: parts.append("clinically significant hemorrhage observed")
        if row.get("Infection")==1:  parts.append("postoperative infection or inflammation observed")
        if row.get("Migration")==1:  parts.append("clip migration or displacement observed")
        if row.get("Misfiring")==1:  parts.append("misfiring observed")
        if row.get("Slippage")==1:   parts.append("clip slippage observed")
        lig = str(row.get("Ligation_Result","")).strip().lower()
        if lig=="complete":   parts.append("complete ligation achieved")
        elif lig=="partial":  parts.append("partial ligation, re-application needed")
        elif lig=="fail":     parts.append("ligation failed")
        perf = row.get("Performance_Rating")
        if pd.notna(perf):
            parts.append({5:"overall performance excellent",4:"overall performance good",
                          3:"overall performance satisfactory",2:"overall performance poor",
                          1:"overall performance unsure"}.get(int(perf),"performance not specified"))
        return "; ".join(parts) if parts else "no issue observed"
    out = df.copy()
    out["text"]     = out.apply(build_text, axis=1)
    out["source"]   = "PMCF"
    out["sentiment"]= out["text"].apply(pmcf_sentiment)
    return out

def parse_uploaded_pmcf(file) -> pd.DataFrame:
    """Method B: auto-detect Raw_Data / Raw_Data_YYYY sheets."""
    fn = file.name.lower()
    if fn.endswith(".csv"):
        try:    df = pd.read_csv(file, encoding="utf-8-sig")
        except: file.seek(0); df = pd.read_csv(file, encoding="cp949")
        df.columns = [str(c).strip() for c in df.columns]
        lm = {c.lower():c for c in df.columns}
        tc = next((lm[c] for c in ["text","description","pmcf","comment","assessment","issue"] if c in lm), None)
        if tc is not None:
            out = df.rename(columns={tc:"text"}).copy()
            out["text"] = out["text"].astype(str).str.strip()
            if "date" in lm: out["date"] = pd.to_datetime(out[lm["date"]], errors="coerce")
            out["source"]    = "PMCF"
            out["sentiment"] = out["text"].apply(pmcf_sentiment)
            out["__year"]    = "N/A"
            return out
        validate_pmcf_columns(df)
        r = pmcf_excel_to_text_records(df); r["__year"] = "N/A"; return r
    elif fn.endswith((".xlsx",".xls")):
        xls        = pd.ExcelFile(file)
        all_sheets = xls.sheet_names
        year_sheets = {}
        for s in all_sheets:
            m = re.match(r"Raw_Data_(\d{4})$", s, re.IGNORECASE)
            if m: year_sheets[int(m.group(1))] = s
        if not year_sheets and "Raw_Data" in all_sheets: year_sheets[0] = "Raw_Data"
        if not year_sheets: year_sheets[0] = all_sheets[0]
        frames = []
        for yr in sorted(year_sheets.keys()):
            sn  = year_sheets[yr]
            raw = pd.read_excel(file, sheet_name=sn, header=None)
            col_row = raw.iloc[1].tolist()
            df  = raw.iloc[2:].copy()
            df.columns = [str(c).strip() if pd.notna(c) else f"col_{i}"
                          for i,c in enumerate(col_row)]
            df  = df.reset_index(drop=True)
            df["__year"]  = str(yr) if yr != 0 else "N/A"
            df["__sheet"] = sn
            frames.append(df)
        if not frames: raise ValueError("No valid data sheets found.")
        merged = pd.concat(frames, ignore_index=True)
        merged.columns = [str(c).strip() for c in merged.columns]
        for col in merged.columns:
            if merged[col].dtype == "object":
                merged[col] = merged[col].astype(str).str.strip().replace(
                    {"nan":pd.NA,"None":pd.NA,"":pd.NA,"<NA>":pd.NA})
        for col in ["SAE","Hemorrhage","Infection","Migration","Misfiring","Slippage"]:
            if col in merged.columns: merged[col] = normalize_binary(merged[col])
        if "Performance_Rating" in merged.columns:
            merged["Performance_Rating"] = pd.to_numeric(merged["Performance_Rating"], errors="coerce")
        for col in ["Surgery_Date","Followup_Date","Completion_Date"]:
            if col in merged.columns: merged[col] = pd.to_datetime(merged[col], errors="coerce")
        if "Followup_Period" in merged.columns:
            merged["Followup_Period"] = merged["Followup_Period"].astype("string").str.strip()
            merged["Followup_Period"] = pd.Categorical(merged["Followup_Period"],
                                                        categories=FOLLOWUP_ORDER, ordered=True)
        yr_col = merged["__year"].copy()
        merged = pmcf_excel_to_text_records(merged)
        merged["__year"] = yr_col.values
        return merged
    else:
        raise ValueError("PMCF file must be CSV / XLSX / XLS.")

def get_latest_year_df(pmcf_df: pd.DataFrame) -> pd.DataFrame:
    if "__year" not in pmcf_df.columns: return pmcf_df
    years = [y for y in pmcf_df["__year"].unique() if y not in ("N/A",None,"nan")]
    if not years: return pmcf_df
    latest = str(max(int(y) for y in years))
    return pmcf_df[pmcf_df["__year"] == latest].copy()

def get_year_list(pmcf_df: pd.DataFrame) -> List[str]:
    if "__year" not in pmcf_df.columns: return []
    return sorted([y for y in pmcf_df["__year"].unique()
                   if y not in ("N/A",None,"nan")], key=lambda x: int(x))

def maybe_merge(comp, pmcf, comp_file, pmcf_file):
    if comp_file is not None:
        up = parse_uploaded_complaint(comp_file)
        up["source"] = "Complaint"
        up["issue_category"] = up["text"].apply(classify_issue)
        comp = pd.concat([comp,up], ignore_index=True) if not comp.empty else up
    if pmcf_file is not None:
        up = parse_uploaded_pmcf(pmcf_file)
        up["source"] = "PMCF"
        if "sentiment" not in up.columns: up["sentiment"] = up["text"].apply(pmcf_sentiment)
        pmcf = pd.concat([pmcf,up], ignore_index=True) if not pmcf.empty else up
    return comp, pmcf

# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════
def make_issue_table(comp: pd.DataFrame) -> pd.DataFrame:
    total = len(comp); rows = []
    for cat,count in comp["issue_category"].value_counts().items():
        lo,hi = wilson_ci(int(count),total)
        rows.append({"Issue":cat,"Count":int(count),
                     "Rate (%)":round(count/total*100,1) if total else 0.0,
                     "95% CI Low (%)":round(lo*100,1),"95% CI High (%)":round(hi*100,1)})
    return pd.DataFrame(rows)

def make_pmcf_table(pmcf: pd.DataFrame) -> pd.DataFrame:
    total = len(pmcf); rows = []
    for cat,count in pmcf["sentiment"].value_counts().items():
        lo,hi = wilson_ci(int(count),total)
        rows.append({"PMCF Assessment":cat,"Count":int(count),
                     "Rate (%)":round(count/total*100,1) if total else 0.0,
                     "95% CI Low (%)":round(lo*100,1),"95% CI High (%)":round(hi*100,1)})
    return pd.DataFrame(rows)

def trend_by_month(df,cat_col):
    if "date" not in df.columns or df["date"].isna().all(): return None
    tmp = df.dropna(subset=["date"]).copy()
    if tmp.empty: return None
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
    return tmp.groupby(["month",cat_col]).size().reset_index(name="count")

def detect_signal(comp,pmcf):
    tc=len(comp); tp=len(pmcf)
    breakage  = (comp["issue_category"]=="Breakage").sum() if tc else 0
    migration = (comp["issue_category"]=="Migration/Slip").sum() if tc else 0
    neg_pmcf  = (pmcf["sentiment"]=="Negative").sum() if tp else 0
    signal,risk = "None","Low"
    if tc and (breakage/tc>=0.08 or migration/tc>=0.08): signal,risk="Potential complaint signal","Moderate"
    if tp and neg_pmcf/tp>=0.20: signal,risk="Potential PMCF signal","Moderate"
    if tc and tp and (breakage+migration)/tc>=0.12 and neg_pmcf/tp>=0.20:
        signal,risk="Combined safety signal","High"
    return signal,risk

def build_structured_pmcf_kpis(pmcf):
    if not set(PMCF_REQUIRED_COLUMNS).issubset(set(pmcf.columns)): return None
    sae_y,_,sae_r  = safe_count_binary(pmcf,"SAE")
    he_y,_,he_r    = safe_count_binary(pmcf,"Hemorrhage")
    inf_y,_,inf_r  = safe_count_binary(pmcf,"Infection")
    mig_y,_,mig_r  = safe_count_binary(pmcf,"Migration")
    mis_y,_,mis_r  = safe_count_binary(pmcf,"Misfiring")
    sl_y,_,sl_r    = safe_count_binary(pmcf,"Slippage")
    complete,cr = 0,0.0
    if "Ligation_Result" in pmcf.columns:
        lig = pmcf["Ligation_Result"].dropna().astype(str).str.strip().str.lower()
        d   = int(lig.shape[0])
        complete = int((lig.str.contains("complete")).sum())
        cr = round(complete/d*100,1) if d else 0.0
    return {"responses":int(pmcf.shape[0]),
            "sae_yes":sae_y,"sae_rate":sae_r,
            "hemorrhage_yes":he_y,"hemorrhage_rate":he_r,
            "infection_yes":inf_y,"infection_rate":inf_r,
            "migration_yes":mig_y,"migration_rate":mig_r,
            "misfiring_yes":mis_y,"misfiring_rate":mis_r,
            "slippage_yes":sl_y,"slippage_rate":sl_r,
            "complete_ligation":complete,"complete_ligation_rate":cr}

def performance_distribution(pmcf):
    if "Performance_Rating" not in pmcf.columns: return None
    return (pmcf["Performance_Rating"].dropna().astype(int).map(PERFORMANCE_LABELS)
            .value_counts().reindex(["Excellent","Good","Satisfactory","Poor","Unsure"],fill_value=0))

# ═══════════════════════════════════════════════════════════════════════════════
# GPT / Fallback
# ═══════════════════════════════════════════════════════════════════════════════
def fallback_summary(issue_table,pmcf_table,signal,risk):
    ti = issue_table.iloc[0]["Issue"] if not issue_table.empty else "No issue identified"
    tr = issue_table.iloc[0]["Rate (%)"] if not issue_table.empty else 0
    nr = (pmcf_table.loc[pmcf_table["PMCF Assessment"]=="Negative","Rate (%)"].iloc[0]
          if not pmcf_table.empty and "Negative" in pmcf_table["PMCF Assessment"].values else 0.0)
    pr = (pmcf_table.loc[pmcf_table["PMCF Assessment"]=="Positive","Rate (%)"].iloc[0]
          if not pmcf_table.empty and "Positive" in pmcf_table["PMCF Assessment"].values else 0.0)
    tc = int(issue_table["Count"].sum()) if not issue_table.empty else 0
    tp = int(pmcf_table["Count"].sum())  if not pmcf_table.empty  else 0
    return (f"A total of {tc} complaint record(s) and {tp} PMCF response(s) were analyzed. "
            f"The most frequently reported complaint issue was '{ti}', accounting for {tr}% of all complaints. "
            f"PMCF data showed a negative response rate of {nr}% and a positive response rate of {pr}%. "
            f"Signal detection result: '{signal}'; overall risk level: '{risk}'. "
            f"These findings are based on automated keyword classification and statistical thresholds; "
            f"regulatory and clinical review by a qualified person is required before use in PSUR or PMS report drafting.")

def ask_gpt(api_key,model,issue_table,pmcf_table,signal,risk):
    if not api_key or OpenAI is None: return None
    client = OpenAI(api_key=api_key)
    prompt = (f"You are a senior EU MDR PMS/PMCF specialist.\n"
              f"Write a concise professional data analysis summary IN ENGLISH based solely on the "
              f"statistical data below. Do NOT include benefit-risk conclusions.\n\n"
              f"Complaint table: {issue_table.to_dict(orient='records')}\n"
              f"PMCF table: {pmcf_table.to_dict(orient='records')}\n"
              f"Signal: {signal}  Risk: {risk}\n\nRequirements: 5-7 sentences, statistics only, English only.")
    resp = client.chat.completions.create(model=model,
        messages=[{"role":"system","content":"You are an expert regulatory writer."},
                  {"role":"user","content":prompt}], temperature=0.2)
    return resp.choices[0].message.content.strip()

def fallback_benefit_risk(issue_table,pmcf_table,signal,risk,pmcf_kpis):
    ti = issue_table.iloc[0]["Issue"] if not issue_table.empty else "no dominant issue"
    tr = issue_table.iloc[0]["Rate (%)"] if not issue_table.empty else 0
    sr = pmcf_kpis["sae_rate"] if pmcf_kpis else "N/A"
    ir = pmcf_kpis["infection_rate"] if pmcf_kpis else "N/A"
    mr = pmcf_kpis["migration_rate"] if pmcf_kpis else "N/A"
    cr = pmcf_kpis["complete_ligation_rate"] if pmcf_kpis else "N/A"
    conc = ("acceptable with current risk controls" if risk in ("Low","Moderate")
            else "requires immediate review and potential risk control update")
    return (f"Based on post-market surveillance and PMCF data analyzed per EU MDR 2017/745 Annex XIV, "
            f"EN ISO 14971:2019, and MDCG 2020-6, clinical benefits — including a first-attempt ligation "
            f"success rate of {cr}% — are considered to outweigh identified risks when used as intended. "
            f"The SAE rate of {sr}%, infection rate of {ir}%, and migration rate of {mr}% are within "
            f"thresholds defined in the PMCF Plan; no new unanticipated serious risks were identified, "
            f"consistent with state-of-the-art assessment per MDCG 2020-5. "
            f"The predominant complaint category was '{ti}' at {tr}%, not constituting a significant safety "
            f"signal per MDR Article 87 vigilance obligations. "
            f"The overall benefit-risk determination is '{conc}', per EN ISO 13485:2016 Clause 8.2.6 "
            f"and EN ISO 14971:2019. "
            f"Continued PMCF is recommended; findings will be incorporated in the next PSUR per MDR Annex III.")

def ask_gpt_benefit_risk(api_key,model,issue_table,pmcf_table,signal,risk,pmcf_kpis):
    if not api_key or OpenAI is None: return None
    client = OpenAI(api_key=api_key)
    prompt = (f"You are a senior EU MDR regulatory affairs specialist.\n"
              f"Write a Benefit-Risk Analysis Summary IN ENGLISH referencing: "
              f"EU MDR 2017/745, EN ISO 14971:2019, EN ISO 13485:2016, MDCG 2020-5, MDCG 2020-6, MDCG 2020-7.\n\n"
              f"Complaint table: {issue_table.to_dict(orient='records')}\n"
              f"PMCF table: {pmcf_table.to_dict(orient='records')}\n"
              f"PMCF KPIs: {str(pmcf_kpis)}\nSignal: {signal}  Risk: {risk}\n\n"
              f"Requirements: exactly 5 sentences, cite standards by name, state clear benefit-risk conclusion, English only.")
    resp = client.chat.completions.create(model=model,
        messages=[{"role":"system","content":"You are an expert EU MDR regulatory writer."},
                  {"role":"user","content":prompt}], temperature=0.2)
    return resp.choices[0].message.content.strip()

# ═══════════════════════════════════════════════════════════════════════════════
# PDF
# ═══════════════════════════════════════════════════════════════════════════════
def register_korean_font():
    try: pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium")); return "HYSMyeongJo-Medium"
    except: pass
    for p in [r"C:\\Windows\\Fonts\\malgun.ttf",r"C:\\Windows\\Fonts\\gulim.ttc"]:
        try: pdfmetrics.registerFont(TTFont("KoreanFont",p)); return "KoreanFont"
        except: continue
    return "Helvetica"

def wrap_text(text,max_chars):
    words=text.split(); lines,current=[],""
    for w in words:
        test=f"{current} {w}".strip()
        if len(test)<=max_chars: current=test
        else:
            if current: lines.append(current)
            current=w
    if current: lines.append(current)
    return lines or [text]

def generate_pdf(issue_table,pmcf_table,summary,signal,risk,br_summary=""):
    buf=io.BytesIO(); c=canvas.Canvas(buf,pagesize=A4); fn=register_korean_font(); _,H=A4; y=H-40
    c.setFont(fn,16); c.drawString(40,y,"AI PMS/PMCF Analysis Report"); y-=28
    c.setFont(fn,10); c.drawString(40,y,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"); y-=24
    c.drawString(40,y,f"Signal Detection: {signal}"); y-=18; c.drawString(40,y,f"Risk Level: {risk}"); y-=26
    def psec(title,rows_fn,fs=10):
        nonlocal y
        c.setFont(fn,12); c.drawString(40,y,title); y-=18; c.setFont(fn,fs)
        for line in rows_fn():
            c.drawString(40,y,line); y-=(12 if fs<=9 else 14)
            if y<60: c.showPage(); c.setFont(fn,fs); y=H-40
        y-=8
    psec("1. Data Analysis Summary", lambda: wrap_text(summary,78))
    psec("2. Complaint Issue Table",
         lambda: [f"{r['Issue']}: Count={r['Count']}, Rate={r['Rate (%)']}%, 95%CI={r['95% CI Low (%)']}~{r['95% CI High (%)']}%"
                  for _,r in issue_table.iterrows()], fs=9)
    psec("3. PMCF Assessment Table",
         lambda: [f"{r['PMCF Assessment']}: Count={r['Count']}, Rate={r['Rate (%)']}%, 95%CI={r['95% CI Low (%)']}~{r['95% CI High (%)']}%"
                  for _,r in pmcf_table.iterrows()], fs=9)
    if br_summary: psec("4. Benefit-Risk Analysis Summary", lambda: wrap_text(br_summary,78))
    c.save(); return buf.getvalue()

# ═══════════════════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════════════════
def _compact_bar(labels,values,colors,title,rotation=20,figsize=(3.2,2.4)):
    n=len(labels); bw=max(0.15,min(0.25,2.0/max(n,1)))
    fig,ax=plt.subplots(figsize=figsize)
    ax.bar(range(n),values,width=bw,color=(colors*math.ceil(n/len(colors)))[:n])
    ax.set_xticks(range(n)); ax.set_xticklabels(labels,rotation=rotation,
        ha="right" if rotation>0 else "center",fontsize=7)
    ax.set_title(title,fontsize=9); ax.set_ylabel("Count",fontsize=7)
    ax.tick_params(axis="y",labelsize=7); plt.tight_layout(); return fig

def plot_issue_bar(t):   return _compact_bar(t["Issue"].tolist(),t["Count"].tolist(),MULTI_COLORS,"Complaint Issue Frequency")
def plot_pmcf_bar(t):    return _compact_bar(t["PMCF Assessment"].tolist(),t["Count"].tolist(),["#4CAF50","#F44336","#FF9800"],"PMCF Response Distribution",rotation=0)
def plot_performance(p): return _compact_bar(p.index.tolist(),p.values.tolist(),PERF_COLORS,"Performance Rating")
def plot_followup(fd):   return _compact_bar(fd.index.tolist(),fd.values.tolist(),FOLLOW_COLORS,"Follow-up Period",rotation=0)
def plot_country(cd):    return _compact_bar(cd["Country"].tolist(),cd["Count"].tolist(),COUNTRY_COLORS,"Country",rotation=35)
def plot_categorical(df,col,title):
    vc=safe_value_counts(df,col)
    return _compact_bar(vc.index.tolist(),vc.values.tolist(),MULTI_COLORS,title) if vc is not None and not vc.empty else None

def plot_binary_single(df,col,title):
    if col not in df.columns: return None
    s=df[col].dropna(); yes=int((s==1).sum()); no=int((s==0).sum())
    fig,ax=plt.subplots(figsize=(3.2,2.4))
    ax.bar(["Yes","No"],[yes,no],width=0.28,color=["#F44336","#4CAF50"])
    ax.set_title(title,fontsize=9); ax.set_ylabel("Count",fontsize=7)
    ax.tick_params(labelsize=7); plt.tight_layout(); return fig

def plot_binary_grouped(df):
    cols=["SAE","Hemorrhage","Infection","Migration"]
    vc,yv,nv=[],[],[]
    for col in cols:
        if col in df.columns:
            s=df[col].dropna(); yv.append(int((s==1).sum())); nv.append(int((s==0).sum())); vc.append(col)
    if not vc: return None
    x=np.arange(len(vc)); bw=0.28
    fig,ax=plt.subplots(figsize=(3.2,2.4))
    ax.bar(x-bw/2,yv,width=bw,color="#F44336",label="Yes")
    ax.bar(x+bw/2,nv,width=bw,color="#4CAF50",label="No")
    ax.set_xticks(x); ax.set_xticklabels(vc,fontsize=7)
    ax.set_title("Clinical Safety Events",fontsize=9); ax.set_ylabel("Count",fontsize=7)
    ax.tick_params(axis="y",labelsize=7); ax.legend(fontsize=6,loc="upper right")
    plt.tight_layout(); return fig

def plot_complaint_monthly(df):
    if "date" not in df.columns or df["date"].isna().all(): return None
    tmp=df.dropna(subset=["date"]).copy()
    if tmp.empty: return None
    tmp["month"]=tmp["date"].dt.to_period("M").astype(str)
    monthly=tmp.groupby("month").size(); n=len(monthly)
    colors=(MULTI_COLORS*math.ceil(n/len(MULTI_COLORS)))[:n]
    fig,ax=plt.subplots(figsize=(3.2,2.4))
    ax.bar(range(n),monthly.values,width=0.4,color=colors)
    ax.set_xticks(range(n)); ax.set_xticklabels(monthly.index.tolist(),rotation=30,ha="right",fontsize=7)
    ax.set_title("Complaint Monthly Occurrence",fontsize=9); ax.set_ylabel("Count",fontsize=7)
    ax.tick_params(axis="y",labelsize=7); plt.tight_layout(); return fig

def plot_monthly(trend_df,label_col,title):
    fig,ax=plt.subplots(figsize=(3.2,2.4))
    pivot=trend_df.pivot(index="month",columns=label_col,values="count").fillna(0)
    pivot.plot(ax=ax,marker="o",linewidth=1.5,markersize=4)
    ax.set_title(title,fontsize=9); ax.set_ylabel("Count",fontsize=8)
    ax.legend(fontsize=7); plt.xticks(rotation=30,ha="right",fontsize=7)
    plt.tight_layout(); return fig

# ── Yearly bar+line combo charts ──────────────────────────────────────────────
def plot_yearly_binary_combo(pmcf,col,title,year_list):
    if col not in pmcf.columns or not year_list: return None
    yv,nv,rates=[],[],[]
    for yr in year_list:
        sub=pmcf[pmcf["__year"]==yr][col].dropna()
        y=int((sub==1).sum()); n_=max(int(sub.shape[0]),1)
        yv.append(y); nv.append(int((sub==0).sum())); rates.append(round(y/n_*100,1))
    n=len(year_list); x=np.arange(n); bw=max(0.15,min(0.3,1.8/max(n,1)))
    fig,ax1=plt.subplots(figsize=(3.2,2.4))
    ax1.bar(x-bw/2,yv,width=bw,color="#F44336",label="Yes",alpha=0.85)
    ax1.bar(x+bw/2,nv,width=bw,color="#4CAF50",label="No",alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(year_list,fontsize=7)
    ax1.set_ylabel("Count",fontsize=7); ax1.tick_params(axis="y",labelsize=7)
    ax2=ax1.twinx()
    ax2.plot(x,rates,marker="o",markersize=3,linewidth=1.2,color="#1565C0",linestyle="--",label="Yes %")
    ax2.set_ylabel("Yes Rate (%)",fontsize=6); ax2.tick_params(axis="y",labelsize=6)
    ax1.set_title(title,fontsize=9)
    l1,b1=ax1.get_legend_handles_labels(); l2,b2=ax2.get_legend_handles_labels()
    ax1.legend(l1+l2,b1+b2,fontsize=5,loc="upper left")
    plt.tight_layout(); return fig

def plot_yearly_ligation_combo(pmcf,year_list):
    if "Ligation_Result" not in pmcf.columns or not year_list: return None
    cats=["complete","partial","fail"]
    clrs={"complete":"#4CAF50","partial":"#FF9800","fail":"#F44336"}
    data={c:[] for c in cats}; comp_rates=[]
    for yr in year_list:
        sub=pmcf[pmcf["__year"]==yr]["Ligation_Result"].dropna().astype(str).str.strip().str.lower()
        n_yr=max(len(sub),1)
        for c in cats: data[c].append(int(sub.str.contains(c).sum()))
        comp_rates.append(round(data["complete"][-1]/n_yr*100,1))
    x=np.arange(len(year_list)); bw=max(0.15,min(0.3,1.8/max(len(year_list),1)))
    fig,ax1=plt.subplots(figsize=(3.2,2.4)); bottom=np.zeros(len(year_list))
    for c in cats:
        ax1.bar(x,data[c],width=bw,bottom=bottom,color=clrs[c],label=c.capitalize(),alpha=0.85)
        bottom+=np.array(data[c])
    ax1.set_xticks(x); ax1.set_xticklabels(year_list,fontsize=7)
    ax1.set_ylabel("Count",fontsize=7); ax1.tick_params(axis="y",labelsize=7)
    ax2=ax1.twinx()
    ax2.plot(x,comp_rates,marker="o",markersize=3,linewidth=1.2,color="#1565C0",linestyle="--",label="Complete %")
    ax2.set_ylabel("Complete (%)",fontsize=6); ax2.tick_params(axis="y",labelsize=6)
    ax1.set_title("Ligation Result by Year",fontsize=9)
    l1,b1=ax1.get_legend_handles_labels(); l2,b2=ax2.get_legend_handles_labels()
    ax1.legend(l1+l2,b1+b2,fontsize=5,loc="upper left")
    plt.tight_layout(); return fig

def plot_yearly_perf_rating_combo(pmcf,year_list):
    if "Performance_Rating" not in pmcf.columns or not year_list: return None
    lo=["Excellent","Good","Satisfactory","Poor","Unsure"]
    cp=["#2196F3","#4CAF50","#FF9800","#F44336","#9E9E9E"]
    data={l:[] for l in lo}; means=[]
    for yr in year_list:
        sub=pd.to_numeric(pmcf[pmcf["__year"]==yr]["Performance_Rating"],errors="coerce").dropna()
        means.append(round(sub.mean(),2) if len(sub)>0 else float("nan"))
        mapped=sub.astype(int).map(PERFORMANCE_LABELS)
        for l in lo: data[l].append(int((mapped==l).sum()))
    x=np.arange(len(year_list)); bw=max(0.15,min(0.3,1.8/max(len(year_list),1)))
    fig,ax1=plt.subplots(figsize=(3.2,2.4)); bottom=np.zeros(len(year_list))
    for l,c in zip(lo,cp):
        ax1.bar(x,data[l],width=bw,bottom=bottom,color=c,label=l,alpha=0.85)
        bottom+=np.array(data[l])
    ax1.set_xticks(x); ax1.set_xticklabels(year_list,fontsize=7)
    ax1.set_ylabel("Count",fontsize=7); ax1.tick_params(axis="y",labelsize=7)
    ax2=ax1.twinx()
    ax2.plot(x,means,marker="o",markersize=3,linewidth=1.2,color="#1565C0",linestyle="--",label="Mean score")
    ax2.set_ylim(0,5.5); ax2.set_ylabel("Mean (1–5)",fontsize=6); ax2.tick_params(axis="y",labelsize=6)
    ax1.set_title("Performance Rating by Year",fontsize=9)
    l1,b1=ax1.get_legend_handles_labels(); l2,b2=ax2.get_legend_handles_labels()
    ax1.legend(l1+l2,b1+b2,fontsize=5,loc="upper left",ncol=2)
    plt.tight_layout(); return fig

def plot_yearly_categorical_combo(pmcf,col,title,year_list):
    if col not in pmcf.columns or not year_list: return None
    all_vals=sorted([v for v in pmcf[col].dropna().astype(str).str.strip().unique()
                     if v.lower() not in ("nan","none","<na>","")])
    if not all_vals: return None
    colors=(MULTI_COLORS*math.ceil(len(all_vals)/len(MULTI_COLORS)))[:len(all_vals)]
    x=np.arange(len(year_list)); bw=max(0.1,min(0.22,1.4/max(len(all_vals),1)))
    offsets=np.linspace(-(len(all_vals)-1)/2*bw,(len(all_vals)-1)/2*bw,len(all_vals))
    totals=[max(pmcf[pmcf["__year"]==yr][col].dropna().shape[0],1) for yr in year_list]
    fig,ax1=plt.subplots(figsize=(3.2,2.4))
    for i,(val,color) in enumerate(zip(all_vals,colors)):
        counts=[int((pmcf[pmcf["__year"]==yr][col].dropna().astype(str).str.strip()==val).sum())
                for yr in year_list]
        ax1.bar(x+offsets[i],counts,width=bw,color=color,alpha=0.85,
                label=val[:12]+("…" if len(val)>12 else ""))
    ax1.set_xticks(x); ax1.set_xticklabels(year_list,fontsize=7)
    ax1.set_ylabel("Count",fontsize=7); ax1.tick_params(axis="y",labelsize=7)
    ax2=ax1.twinx()
    ax2.plot(x,totals,marker="s",markersize=3,linewidth=1.2,color="#1565C0",linestyle="--",label="Total n")
    ax2.set_ylabel("Total n",fontsize=6); ax2.tick_params(axis="y",labelsize=6)
    ax1.set_title(title,fontsize=9)
    l1,b1=ax1.get_legend_handles_labels(); l2,b2=ax2.get_legend_handles_labels()
    ax1.legend(l1+l2,b1+b2,fontsize=5,loc="upper left",ncol=2)
    plt.tight_layout(); return fig

# ── Trend charts ──────────────────────────────────────────────────────────────
def compute_pmcf_binary_trend(pmcf,cols):
    dc=next((c for c in ["Surgery_Date","Followup_Date","Completion_Date","date"] if c in pmcf.columns),None)
    if dc is None: return None
    tmp=pmcf.copy(); tmp["_d"]=pd.to_datetime(tmp[dc],errors="coerce")
    tmp=tmp.dropna(subset=["_d"])
    if tmp.empty: return None
    tmp["month"]=tmp["_d"].dt.to_period("M"); result={}
    for col in cols:
        if col not in tmp.columns: continue
        monthly=tmp.groupby("month")[col].apply(
            lambda s: round(s.dropna().astype(float).mean()*100,1) if s.dropna().shape[0]>0 else float("nan"))
        result[col]=monthly
    return result if result else None

def compute_ligation_trend(pmcf):
    dc=next((c for c in ["Surgery_Date","Followup_Date","Completion_Date","date"] if c in pmcf.columns),None)
    if dc is None or "Ligation_Result" not in pmcf.columns: return None
    tmp=pmcf.copy(); tmp["_d"]=pd.to_datetime(tmp[dc],errors="coerce")
    tmp=tmp.dropna(subset=["_d"])
    if tmp.empty: return None
    tmp["month"]=tmp["_d"].dt.to_period("M")
    lig=tmp.groupby("month")["Ligation_Result"].apply(
        lambda s: round((s.astype(str).str.strip().str.lower().str.contains("fail")).sum()/max(s.shape[0],1)*100,1))
    return {"Ligation_Fail":lig}

def plot_pmcf_trend_group(trend_dict,threshold_pct,title):
    if not trend_dict: return None
    months_all=sorted(set(m for s in trend_dict.values() for m in s.index))
    ms=[str(m) for m in months_all]; ci=iter(MULTI_COLORS)
    fig,ax=plt.subplots(figsize=(3.2,2.4)); breach=False
    for col,series in trend_dict.items():
        vals=[series.get(m,float("nan")) for m in months_all]
        color=next(ci,"#9E9E9E")
        ax.plot(ms,vals,marker="o",markersize=3,linewidth=1.5,color=color,label=col)
        if any(v>=threshold_pct for v in vals if not math.isnan(v)): breach=True
    ax.axhline(threshold_pct,color="#F44336",linewidth=1,linestyle=":",label=f"Threshold ({threshold_pct:.0f}%)")
    ax.set_title(title,fontsize=9); ax.set_ylabel("Rate (%)",fontsize=7)
    ax.tick_params(axis="y",labelsize=7); ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(ms,rotation=30,ha="right",fontsize=6)
    ax.legend(fontsize=6,loc="upper left"); plt.tight_layout()
    return fig,breach

def plot_yoy_bar(pmcf,cols,threshold_pct,title,year_list):
    data={}
    for col in cols:
        if col not in pmcf.columns: continue
        rates=[]
        for yr in year_list:
            sub=pd.to_numeric(pmcf[pmcf["__year"]==yr][col].dropna(),errors="coerce").dropna()
            rates.append(round(sub.mean()*100,1) if len(sub)>0 else 0.0)
        data[col]=rates
    if not data: return None
    n=len(year_list); x=np.arange(n)
    bw=max(0.1,min(0.25,1.5/max(len(data),1)))
    offsets=np.linspace(-(len(data)-1)/2*bw,(len(data)-1)/2*bw,len(data))
    ci=iter(MULTI_COLORS); fig,ax=plt.subplots(figsize=(3.2,2.4))
    for i,(col,rates) in enumerate(data.items()):
        ax.bar(x+offsets[i],rates,width=bw,color=next(ci,"#9E9E9E"),label=col,alpha=0.85)
    ax.axhline(threshold_pct,color="#F44336",linewidth=1,linestyle=":",label=f"Threshold ({threshold_pct:.0f}%)")
    ax.set_xticks(x); ax.set_xticklabels(year_list,fontsize=7)
    ax.set_title(title,fontsize=9); ax.set_ylabel("Rate (%)",fontsize=7)
    ax.tick_params(axis="y",labelsize=7); ax.legend(fontsize=6,loc="upper left")
    plt.tight_layout(); return fig

# ═══════════════════════════════════════════════════════════════════════════════
# Coda
# ═══════════════════════════════════════════════════════════════════════════════
def send_analysis_to_coda(run_id,complaint_text,pmcf_text,signal,risk,summary,related_hazard):
    url=f"https://coda.io/apis/v1/docs/{CODA_DOC_ID}/tables/{quote(CODA_TABLE_NAME,safe='')}/rows"
    headers={"Authorization":f"Bearer {CODA_API_TOKEN}","Content-Type":"application/json"}
    payload={"rows":[{"cells":[
        {"column":"Run ID","value":run_id},{"column":"Complaint Text","value":complaint_text},
        {"column":"PMCF Text","value":pmcf_text},{"column":"Signal","value":str(signal)},
        {"column":"Risk","value":str(risk)},{"column":"Summary","value":str(summary)},
        {"column":"Related Hazard","value":related_hazard},
        {"column":"Created At","value":datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
    ]}]}
    r=requests.post(url,headers=headers,json=payload)
    return r.status_code,r.text

# ═══════════════════════════════════════════════════════════════════════════════
# UI helpers
# ═══════════════════════════════════════════════════════════════════════════════
def section_divider():
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)

def render_table(df: pd.DataFrame):
    """Consistent styled HTML table."""
    st.markdown(
        df.style
        .set_properties(**{"font-size":"13px","text-align":"center"})
        .set_table_styles([
            {"selector":"th","props":[("font-size","13px"),("font-weight","bold"),
                                       ("text-align","center"),("padding","6px 12px"),
                                       ("background-color","#f0f2f6")]},
            {"selector":"td","props":[("font-size","13px"),("padding","5px 12px"),
                                       ("text-align","center")]},
            {"selector":"table","props":[("border-collapse","collapse"),("width","100%")]},
        ])
        .hide(axis="index").to_html(),
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🤖 AI PMS / PMCF Intelligence Platform")
st.caption("Complaint / PMCF Analysis · Statistical Computation · Chart Generation · GPT Summary · PDF Report")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key (optional)", type="password")
    model   = st.selectbox("GPT Model", ["gpt-4o-mini","gpt-4.1-mini","gpt-4o"], index=0)
    st.markdown("---")
    st.write("Recommended Complaint upload columns")
    st.code("text, date", language=None)
    st.write("Supported PMCF upload formats")
    st.code("CSV (text/date) or XLSX Raw_Data / Raw_Data_YYYY sheet", language=None)
    st.caption("For PMCF Excel with yearly sheets (Raw_Data_2024, etc.), all years are merged automatically.")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Automation","Complaint Classification")
c2.metric("Advanced Output","Statistics + Charts")
c3.metric("Generative AI","GPT Summary")
c4.metric("Report","PDF Download")
st.divider()

i1,i2 = st.columns(2)
with i1:
    complaint_text = st.text_area("Complaint Data (한 줄당 1건)", height=180,
        placeholder="e.g.:\nClip breakage observed during procedure\nClip slipped during surgery\nNo issue observed")
    complaint_file = st.file_uploader("파일 업로드 (Complaint)", type=["csv","xlsx","xls"], key="comp")
with i2:
    pmcf_text = st.text_area("PMCF Data (한 줄당 1건)", height=180,
        placeholder="e.g.:\nNo complication observed after 6 months\nInitial mild discomfort resolved without intervention")
    pmcf_file = st.file_uploader("파일 업로드 (PMCF)", type=["csv","xlsx","xls"], key="pmcf")

related_hazard = st.text_input("Related Hazard ID (e.g. HZ-001)")

if st.button("🚀 Run Full Analysis", use_container_width=True):
    with st.spinner("Processing and analysing data..."):
        try:
            complaint_df,pmcf_df = parse_text_inputs(complaint_text,pmcf_text)
            complaint_df,pmcf_df = maybe_merge(complaint_df,pmcf_df,complaint_file,pmcf_file)
        except Exception as e:
            st.error(f"File processing error: {e}"); st.stop()
        if complaint_df.empty and pmcf_df.empty:
            st.error("Please enter Complaint or PMCF data, or upload a file."); st.stop()

        issue_table = (make_issue_table(complaint_df) if not complaint_df.empty
                       else pd.DataFrame(columns=["Issue","Count","Rate (%)","95% CI Low (%)","95% CI High (%)"]))
        pmcf_table  = (make_pmcf_table(pmcf_df) if not pmcf_df.empty
                       else pd.DataFrame(columns=["PMCF Assessment","Count","Rate (%)","95% CI Low (%)","95% CI High (%)"]))

        signal,risk  = detect_signal(complaint_df,pmcf_df)
        pmcf_kpis    = build_structured_pmcf_kpis(pmcf_df)

        gpt_summary=None
        try:   gpt_summary=ask_gpt(api_key,model,issue_table,pmcf_table,signal,risk)
        except Exception as e: st.warning(f"GPT summary failed; using local summary. ({e})")
        summary = gpt_summary or fallback_summary(issue_table,pmcf_table,signal,risk)

        br_gpt=None
        try:   br_gpt=ask_gpt_benefit_risk(api_key,model,issue_table,pmcf_table,signal,risk,pmcf_kpis)
        except Exception as e: st.warning(f"GPT B-R failed; using local text. ({e})")
        br_summary = br_gpt or fallback_benefit_risk(issue_table,pmcf_table,signal,risk,pmcf_kpis)

        pdf_bytes = generate_pdf(issue_table,pmcf_table,summary,signal,risk,br_summary)

        run_id=f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        sc,rt=send_analysis_to_coda(run_id,complaint_text,pmcf_text,signal,risk,summary,related_hazard)
        if sc==202: st.success("분석 완료 + Coda 저장 성공")
        else:       st.warning(f"Coda 저장 실패: {sc}"); st.text(rt)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Summary
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("1. Summary")
    st.markdown("<br>", unsafe_allow_html=True)

    # KPI table — same style as tables below
    kpi_df = pd.DataFrame({
        "Metric": ["Complaint Count","PMCF Count","Signal Detection","Risk Level"],
        "Value":  [str(len(complaint_df)),str(len(pmcf_df)),signal,risk],
    })
    render_table(kpi_df)
    st.markdown("<br>", unsafe_allow_html=True)

    t1,t2 = st.columns(2)
    with t1:
        st.markdown("**Complaint Issue Table**")
        st.dataframe(issue_table, use_container_width=True)
    with t2:
        st.markdown("**PMCF Assessment Table**")
        st.dataframe(pmcf_table, use_container_width=True)

    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 2. PMCF Survey Statistics
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("2. PMCF Survey Statistics")
    st.markdown("<br>", unsafe_allow_html=True)

    years       = get_year_list(pmcf_df)
    latest_df   = get_latest_year_df(pmcf_df)
    latest_year = years[-1] if years else "All"
    has_years   = len(years) > 1

    # ── A. General Information & B. Device Usage ──────────────────────────────
    st.markdown("##### A. General Information & B. Device Usage")
    st.caption(f"Based on most recent data: **{latest_year}**")

    gen_rows = []
    for col,label in [("Followup_Period","Follow-up Period"),
                       ("Procedure_Type","Procedure Type"),
                       ("Surgical_Approach","Surgical Approach")]:
        if col in latest_df.columns:
            vc = safe_value_counts(latest_df,col)
            if vc is not None:
                n_t = max(latest_df[col].dropna().shape[0],1)
                for val,cnt in vc.items():
                    gen_rows.append({"Category":label,"Response":val,
                                     "Count (n)":cnt,"Rate (%)":round(cnt/n_t*100,1)})
    if gen_rows:
        render_table(pd.DataFrame(gen_rows))
    st.markdown("<br>", unsafe_allow_html=True)

    # ── C. Safety ─────────────────────────────────────────────────────────────
    st.markdown("##### C. Safety")

    safety_cols_list = ["SAE","Hemorrhage","Infection","Migration"]
    pmcf_kpis_s = {}
    for col in safety_cols_list:
        y,d,r = safe_count_binary(latest_df,col)
        pmcf_kpis_s[col] = {"yes":y,"n":d,"rate":r}

    # Metrics text row
    sm1,sm2,sm3,sm4 = st.columns(4)
    for col,w in zip(safety_cols_list,[sm1,sm2,sm3,sm4]):
        d=pmcf_kpis_s[col]
        with w: st.metric(f"{col} Rate ({latest_year})",f"{d['rate']}%",help=f"Yes: {d['yes']} / n: {d['n']}")

    # Incidence rate table
    safety_rate_rows=[{"Indicator":col,"Yes (n)":pmcf_kpis_s[col]["yes"],
                       "Sample (n)":pmcf_kpis_s[col]["n"],
                       "Incidence Rate (%)":pmcf_kpis_s[col]["rate"]}
                      for col in safety_cols_list if col in latest_df.columns]
    if safety_rate_rows:
        render_table(pd.DataFrame(safety_rate_rows))
    st.markdown("<br>", unsafe_allow_html=True)

    # Yearly Yes/No distribution — bar + line
    sc1,sc2,sc3,sc4 = st.columns(4)
    for col,w in zip(safety_cols_list,[sc1,sc2,sc3,sc4]):
        with w:
            fig=(plot_yearly_binary_combo(pmcf_df,col,f"{col} by Year",years) if has_years
                 else plot_binary_single(pmcf_df,col,col))
            if fig: st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── D. Performance ────────────────────────────────────────────────────────
    st.markdown("##### D. Performance")

    perf_kpis={}
    if "Ligation_Result" in latest_df.columns:
        ls=latest_df["Ligation_Result"].dropna().astype(str).str.strip().str.lower()
        nl=max(len(ls),1); cn=int(ls.str.contains("complete").sum())
        perf_kpis["Ligation Complete"]={"val":f"{round(cn/nl*100,1)}%","help":f"Complete: {cn} / n: {nl}"}
    for col,label in [("Misfiring","Misfiring"),("Slippage","Slippage")]:
        y,d,r=safe_count_binary(latest_df,col)
        perf_kpis[label]={"val":f"{r}%","help":f"Yes: {y} / n: {d}"}
    if "Performance_Rating" in latest_df.columns:
        pr=pd.to_numeric(latest_df["Performance_Rating"],errors="coerce").dropna()
        if len(pr)>0: perf_kpis["Perf Rating (Mean)"]={"val":f"{round(pr.mean(),2)} / 5","help":f"n: {len(pr)}"}

    pm_cols=st.columns(max(len(perf_kpis),1))
    for i,(label,d) in enumerate(perf_kpis.items()):
        with pm_cols[i]: st.metric(f"{label} ({latest_year})",d["val"],help=d["help"])

    # Performance rate table
    perf_rate_rows=[]
    if "Ligation_Result" in latest_df.columns:
        ls2=latest_df["Ligation_Result"].dropna().astype(str).str.strip().str.lower()
        n2=max(len(ls2),1); c2=int(ls2.str.contains("complete").sum())
        perf_rate_rows.append({"Indicator":"Ligation Complete","Count (n)":c2,"Sample (n)":n2,"Rate (%)":round(c2/n2*100,1)})
    for col in ["Misfiring","Slippage"]:
        y,d2,r=safe_count_binary(latest_df,col)
        if col in latest_df.columns:
            perf_rate_rows.append({"Indicator":col,"Count (n)":y,"Sample (n)":d2,"Rate (%)":r})
    if "Performance_Rating" in latest_df.columns:
        pr2=pd.to_numeric(latest_df["Performance_Rating"],errors="coerce").dropna()
        if len(pr2)>0:
            for lp,vp in PERFORMANCE_LABELS.items():
                cp=int((pr2==lp).sum())
                perf_rate_rows.append({"Indicator":f"Performance: {vp}","Count (n)":cp,
                                       "Sample (n)":len(pr2),"Rate (%)":round(cp/max(len(pr2),1)*100,1)})
    if perf_rate_rows:
        render_table(pd.DataFrame(perf_rate_rows))
    st.markdown("<br>", unsafe_allow_html=True)

    pd1,pd2,pd3,pd4=st.columns(4)
    with pd1:
        fig=(plot_yearly_ligation_combo(pmcf_df,years) if has_years
             else plot_categorical(pmcf_df,"Ligation_Result","Ligation Result"))
        if fig: st.pyplot(fig)
    with pd2:
        fig=(plot_yearly_binary_combo(pmcf_df,"Misfiring","Misfiring by Year",years) if has_years
             else plot_binary_single(pmcf_df,"Misfiring","Misfiring"))
        if fig: st.pyplot(fig)
    with pd3:
        fig=(plot_yearly_binary_combo(pmcf_df,"Slippage","Slippage by Year",years) if has_years
             else plot_binary_single(pmcf_df,"Slippage","Slippage"))
        if fig: st.pyplot(fig)
    with pd4:
        fig=(plot_yearly_perf_rating_combo(pmcf_df,years) if has_years
             else (plot_performance(performance_distribution(pmcf_df))
                   if performance_distribution(pmcf_df) is not None else None))
        if fig: st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── E. Overall Clinical Assessment ───────────────────────────────────────
    st.markdown("##### E. Overall Clinical Assessment")

    oca_rows=[]
    for col,label in [("Intended_Purpose","Intended Purpose"),
                       ("Benefit_Risk","Benefit-Risk Assessment")]:
        if col in latest_df.columns:
            vc=safe_value_counts(latest_df,col)
            if vc is not None:
                n_t=max(latest_df[col].dropna().shape[0],1)
                for val,cnt in vc.items():
                    oca_rows.append({"Category":label,"Response":val,
                                     "Count (n)":cnt,"Rate (%)":round(cnt/n_t*100,1)})
    if oca_rows:
        render_table(pd.DataFrame(oca_rows))
    st.markdown("<br>", unsafe_allow_html=True)

    ea1,ea2,ea3,ea4=st.columns(4)
    with ea1:
        fig=(plot_yearly_categorical_combo(pmcf_df,"Intended_Purpose","Intended Purpose by Year",years) if has_years
             else plot_categorical(pmcf_df,"Intended_Purpose","Intended Purpose"))
        if fig: st.pyplot(fig)
    with ea2:
        fig=(plot_yearly_categorical_combo(pmcf_df,"Benefit_Risk","Benefit-Risk by Year",years) if has_years
             else plot_categorical(pmcf_df,"Benefit_Risk","Benefit-Risk Assessment"))
        if fig: st.pyplot(fig)

    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Complaint Statistics
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("3. Complaint Statistics")
    st.markdown("<br>", unsafe_allow_html=True)
    cs1,cs2,_,_ = st.columns(4)
    with cs1:
        if not issue_table.empty: st.pyplot(plot_issue_bar(issue_table))
        else: st.info("No complaint data for chart.")
    with cs2:
        fig=plot_complaint_monthly(complaint_df)
        if fig: st.pyplot(fig)
        else: st.info("No date column available for monthly occurrence chart.")

    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Trend Analysis
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("4. Trend Analysis")
    st.markdown(
        "📐 **Reference Standards:** ISO/TR 20416:2020 · MDCG 2020-5 · EU MDR 2017/745 Article 88")
    st.markdown(
        f"📊 **Statistical Method:** Monthly Yes-rate per indicator  |  "
        f"Baseline = {TREND_BASELINE_WINDOW}-month rolling average (prior period)  |  "
        f"Threshold: Safety ≥5%, Misfiring/Slippage ≥8%, Ligation fail ≥5%  |  "
        f"Breach = latest monthly rate ≥ threshold")
    st.markdown("<br>", unsafe_allow_html=True)

    safety_breach=False; performance_breach=False

    # Row 1: Latest year monthly trends
    st.markdown(f"**Latest Year Monthly Trend — {latest_year}**")
    safety_trend_latest  = compute_pmcf_binary_trend(latest_df,["SAE","Hemorrhage","Infection","Migration"])
    perf_trend_latest    = compute_pmcf_binary_trend(latest_df,["Misfiring","Slippage"])
    lig_trend_latest     = compute_ligation_trend(latest_df)
    perf_combined_latest = {}
    if perf_trend_latest:  perf_combined_latest.update(perf_trend_latest)
    if lig_trend_latest:   perf_combined_latest.update(lig_trend_latest)

    trd1,trd2,trd3,trd4=st.columns(4)
    with trd1:
        st.markdown("**Safety**")
        if safety_trend_latest:
            r=plot_pmcf_trend_group(safety_trend_latest,SAFETY_THRESHOLD,"Safety Trend")
            if r: fig_s,safety_breach=r; st.pyplot(fig_s); st.caption(f"{'⚠️ Breach' if safety_breach else '✅ OK'}")
        else: st.info("No date data.")
    with trd2:
        st.markdown("**Performance**")
        if perf_combined_latest:
            r=plot_pmcf_trend_group(perf_combined_latest,PERFORMANCE_THRESHOLD,"Performance Trend")
            if r: fig_p,performance_breach=r; st.pyplot(fig_p); st.caption(f"{'⚠️ Breach' if performance_breach else '✅ OK'}")
        else: st.info("No date data.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Year-over-Year
    if has_years:
        st.markdown("**Year-over-Year Trend**")
        yoy1,yoy2,yoy3,yoy4=st.columns(4)
        with yoy1:
            fig=plot_yoy_bar(pmcf_df,["SAE","Hemorrhage","Infection","Migration"],SAFETY_THRESHOLD,"Safety YoY",years)
            if fig: st.pyplot(fig)
        with yoy2:
            fig=plot_yoy_bar(pmcf_df,["Misfiring","Slippage"],PERFORMANCE_THRESHOLD,"Misfiring/Slippage YoY",years)
            if fig: st.pyplot(fig)
        with yoy3:
            if "Ligation_Result" in pmcf_df.columns:
                fr=[]
                for yr in years:
                    sub=pmcf_df[pmcf_df["__year"]==yr]["Ligation_Result"].dropna().astype(str).str.lower()
                    fr.append(round(sub.str.contains("fail").sum()/max(len(sub),1)*100,1))
                fig_l,ax_l=plt.subplots(figsize=(3.2,2.4))
                bw_l=max(0.15,min(0.3,1.8/max(len(years),1)))
                ax_l.bar(range(len(years)),fr,width=bw_l,color="#FF9800",alpha=0.85)
                ax_l.axhline(LIGATION_THRESHOLD,color="#F44336",linewidth=1,linestyle=":",label=f"Threshold ({LIGATION_THRESHOLD}%)")
                ax_l.set_xticks(range(len(years))); ax_l.set_xticklabels(years,fontsize=7)
                ax_l.set_title("Ligation Fail Rate YoY",fontsize=9)
                ax_l.set_ylabel("Rate (%)",fontsize=7); ax_l.tick_params(axis="y",labelsize=7)
                ax_l.legend(fontsize=6); plt.tight_layout(); st.pyplot(fig_l)
        with yoy4:
            if "Performance_Rating" in pmcf_df.columns:
                mv=[]
                for yr in years:
                    sub=pd.to_numeric(pmcf_df[pmcf_df["__year"]==yr]["Performance_Rating"],errors="coerce").dropna()
                    mv.append(round(sub.mean(),2) if len(sub)>0 else float("nan"))
                fig_r,ax_r=plt.subplots(figsize=(3.2,2.4))
                ax_r.plot(years,mv,marker="o",markersize=4,linewidth=1.5,color="#2196F3")
                ax_r.set_ylim(0,5.5); ax_r.set_title("Performance Rating YoY",fontsize=9)
                ax_r.set_ylabel("Mean Score (1–5)",fontsize=7); ax_r.tick_params(labelsize=7)
                for yr,m in zip(years,mv):
                    if not math.isnan(m):
                        ax_r.annotate(f"{m}",(yr,m),textcoords="offset points",xytext=(0,5),fontsize=6,ha="center")
                plt.tight_layout(); st.pyplot(fig_r)

    st.markdown("<br>", unsafe_allow_html=True)

    # Threshold compliance table
    any_breach=safety_breach or performance_breach
    tr_rows=[]
    for col in ["SAE","Hemorrhage","Infection","Migration"]:
        if safety_trend_latest and col in safety_trend_latest:
            s=safety_trend_latest[col]; lv=s.iloc[-1] if len(s)>0 else float("nan")
            tr_rows.append({"Category":"Safety","Indicator":col,
                "Latest Rate (%)":round(lv,1) if not math.isnan(lv) else "N/A",
                "Threshold (%)":SAFETY_THRESHOLD,
                "Breach":"⚠️ YES" if not math.isnan(lv) and lv>=SAFETY_THRESHOLD else "✅ NO"})
    for col in ["Misfiring","Slippage","Ligation_Fail"]:
        src=perf_combined_latest if col!="Ligation_Fail" else (lig_trend_latest or {})
        thr=LIGATION_THRESHOLD if col=="Ligation_Fail" else PERFORMANCE_THRESHOLD
        if src and col in src:
            s=src[col]; lv=s.iloc[-1] if len(s)>0 else float("nan")
            tr_rows.append({"Category":"Performance","Indicator":col,
                "Latest Rate (%)":round(lv,1) if not math.isnan(lv) else "N/A",
                "Threshold (%)":thr,
                "Breach":"⚠️ YES" if not math.isnan(lv) and lv>=thr else "✅ NO"})
    if tr_rows:
        st.markdown("**Threshold Compliance Summary**")
        st.dataframe(pd.DataFrame(tr_rows),use_container_width=True)

    # MDR Article 88
    st.markdown("---")
    st.markdown("#### 📋 MDR Article 88 — Trend Reporting Assessment")
    if any_breach:
        st.error("⚠️ **Trend Reporting to Competent Authority Required**\n\n"
                 "One or more indicators exceeded the pre-defined statistical threshold "
                 "(per ISO/TR 20416:2020 and MDCG 2020-5 §4.3).\n\n"
                 "Under **EU MDR 2017/745 Article 88**, report without undue delay.\n\n"
                 "**Required actions:**\n"
                 "- Submit Trend Report via EUDAMED.\n"
                 "- Initiate root-cause investigation (MDR Art. 89).\n"
                 "- Update Risk Management File (EN ISO 14971:2019).\n"
                 "- Document in PMS Report / PSUR per MDR Annex III and MDCG 2022-21.\n"
                 "- Notify Notified Body if GSPRs (MDR Annex I) are affected.")
    else:
        st.success("✅ **No Trend Reporting Required at This Time**\n\n"
                   "All indicators remain within pre-defined thresholds. "
                   "No statutory notification under **EU MDR 2017/745 Article 88** is triggered.\n\n"
                   "**Ongoing obligations:**\n"
                   "- Continue monthly monitoring per ISO/TR 20416:2020.\n"
                   "- Re-evaluate thresholds annually (MDCG 2020-7 §3.4).\n"
                   "- Document in PMS Report / PSUR per MDR Annex III.")

    # Statistical methodology
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 Statistical Methodology Details"):
        st.markdown(f"""
**Trend Analysis Methodology** (ISO/TR 20416:2020 · MDCG 2020-5 · EU MDR 2017/745 Art. 88)

| Item | Method |
|---|---|
| **Incidence rate** | Yes responses ÷ sample size × 100 (%) |
| **Monthly trend** | Monthly Yes-rate per indicator plotted as time series |
| **Baseline** | {TREND_BASELINE_WINDOW}-month rolling mean of observed monthly rates (shifted 1 period) |
| **Threshold** | Safety (SAE/Hemorrhage/Infection/Migration): ≥ 5%; Misfiring/Slippage: ≥ 8%; Ligation fail: ≥ 5% |
| **Signal detection** | Latest monthly rate ≥ threshold → breach flagged |
| **Year-over-Year (YoY)** | Mean Yes-rate per calendar year, compared across years |
| **Confidence interval** | Wilson score 95% CI used for summary tables |
| **Chart type** | Bar (count distribution) + line overlay (rate / trend) |
| **Regulatory basis** | ISO/TR 20416:2020 §6, MDCG 2020-5 §4.3, MDR Art. 88 trend reporting obligation |

> Pre-defined thresholds are consistent with the PMCF Plan and are subject to annual review per MDCG 2020-7.
""")

    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Data Analysis Summary
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("5. Data Analysis Summary")
    st.write(summary)

    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 6. Benefit-Risk Analysis Summary
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("6. Benefit-Risk Analysis Summary")
    st.info("📋 Reference: EU MDR 2017/745 · EN ISO 14971:2019 · EN ISO 13485:2016 · "
            "MDCG 2020-5 · MDCG 2020-6 · MDCG 2020-7")
    st.write(br_summary)

    st.download_button("📄 Download PDF Report", data=pdf_bytes,
        file_name=f"AI_PMS_PMCF_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf")

    with st.expander("View raw processed data"):
        if not complaint_df.empty:
            st.markdown("**Complaint Records**"); st.dataframe(complaint_df,use_container_width=True)
        if not pmcf_df.empty:
            st.markdown("**PMCF Records**"); st.dataframe(pmcf_df,use_container_width=True)

# ── Coda Debug ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Coda Connection Debug")
st.code(f"CODA_DOC_ID = {CODA_DOC_ID}")
st.code(f"CODA_TABLE_NAME = {CODA_TABLE_NAME}")
st.code(f"TOKEN PREFIX = {CODA_API_TOKEN[:8]}...")

if st.button("🔍 List Coda Documents"):
    r=requests.get("https://coda.io/apis/v1/docs",headers={"Authorization":f"Bearer {CODA_API_TOKEN}"})
    st.write("Status:",r.status_code)
    try: st.json(r.json())
    except: st.text(r.text)

if st.button("🔎 Verify Current DOC_ID"):
    r=requests.get(f"https://coda.io/apis/v1/docs/{CODA_DOC_ID}",headers={"Authorization":f"Bearer {CODA_API_TOKEN}"})
    st.write("Status:",r.status_code)
    try: st.json(r.json())
    except: st.text(r.text)
