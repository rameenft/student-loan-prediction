from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression

from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format, Scheme
import textwrap


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "242A_model_data.csv"

MODEL_DEFAULT_APP_PATH = DATA_DIR / "model_default_app.pkl"
MODEL_HIGHDEBT_APP_PATH = DATA_DIR / "model_high_debt_app.pkl"

COEF_DEFAULT_APP_PATH = DATA_DIR / "dash_logit_coefficients_default_app.csv"
COEF_HIGHDEBT_APP_PATH = DATA_DIR / "dash_logit_coefficients_highdebt_app.csv"
PERM_DEFAULT_APP_PATH = DATA_DIR / "dash_perm_importance_default_app.csv"
PERM_HIGHDEBT_APP_PATH = DATA_DIR / "dash_perm_importance_highdebt_app.csv"

CAL_DEFAULT_APP_PATH = DATA_DIR / "dash_calibration_curve_default_app.csv"
CAL_HIGHDEBT_APP_PATH = DATA_DIR / "dash_calibration_curve_highdebt_app.csv"
ROC_DEFAULT_APP_PATH = DATA_DIR / "dash_roc_default_app.csv"
ROC_HIGHDEBT_APP_PATH = DATA_DIR / "dash_roc_highdebt_app.csv"
PR_DEFAULT_APP_PATH = DATA_DIR / "dash_pr_default_app.csv"
PR_HIGHDEBT_APP_PATH = DATA_DIR / "dash_pr_highdebt_app.csv"
SWEEP_DEFAULT_APP_PATH = DATA_DIR / "dash_threshold_sweep_default_app.csv"
SWEEP_HIGHDEBT_APP_PATH = DATA_DIR / "dash_threshold_sweep_highdebt_app.csv"
OOF_DEFAULT_APP_PATH = DATA_DIR / "dash_oof_predictions_default_app.csv"
OOF_HIGHDEBT_APP_PATH = DATA_DIR / "dash_oof_predictions_highdebt_app.csv"

# Full feature-set diagnostics (comparison charts)
CAL_DEFAULT_FULL_PATH = DATA_DIR / "dash_calibration_curve_default_full.csv"
CAL_HIGHDEBT_FULL_PATH = DATA_DIR / "dash_calibration_curve_highdebt_full.csv"
ROC_DEFAULT_FULL_PATH = DATA_DIR / "dash_roc_default.csv"
ROC_HIGHDEBT_FULL_PATH = DATA_DIR / "dash_roc_highdebt.csv"
PR_DEFAULT_FULL_PATH = DATA_DIR / "dash_pr_default.csv"
PR_HIGHDEBT_FULL_PATH = DATA_DIR / "dash_pr_highdebt.csv"
SWEEP_DEFAULT_FULL_PATH = DATA_DIR / "dash_threshold_sweep_default.csv"
SWEEP_HIGHDEBT_FULL_PATH = DATA_DIR / "dash_threshold_sweep_highdebt.csv"

THR_TUNE_DEFAULT_FULL_PATH = DATA_DIR / "dash_threshold_tuning_default_full_calibrated.csv"
THR_TUNE_HIGHDEBT_FULL_PATH = DATA_DIR / "dash_threshold_tuning_highdebt_full_calibrated.csv"

# PCA
PCA_EMBEDDINGS_PATH = DATA_DIR / "dash_pca_embeddings.csv"
PCA_VAR_PATH = DATA_DIR / "dash_pca_explained_variance.csv"
CLUSTER_SUMMARY_PATH = DATA_DIR / "dash_cluster_summary.csv"

MODEL_PERF_PATH = DATA_DIR / "dash_model_performance.csv"

# ------------------------------------------------------------
# CSS (write to assets/styles.css)
# ------------------------------------------------------------
CSS_PATH = ASSETS_DIR / "styles.css"
if not CSS_PATH.exists():
    CSS_PATH.write_text(
        """
:root{
  --bg:#ffffff;
  --text:#111827;
  --muted:#6b7280;
  --border:#e5e7eb;
  --card:#ffffff;
  --shadow:0 1px 2px rgba(0,0,0,0.06);
  --radius:14px;
  --gap:14px;
  --maxw:1240px;
  --font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,"Apple Color Emoji","Segoe UI Emoji";
  --brand:#2563eb;
  --brand2:#1d4ed8;
  --pill:#f8fafc;
}

html, body {
  height: 100%;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  margin: 0;
}

#_dash-app-content { min-height: 100vh; }

.container {
  max-width: var(--maxw);
  margin: 0 auto;
  padding: 18px 18px 44px 18px;
}

.header {
  display:flex;
  flex-direction:column;
  gap:6px;
  align-items:center;
  margin-bottom:14px;
}

.subtle { color: var(--muted); font-size:0.95rem; }

.navbar {
  display:flex;
  flex-wrap:wrap;
  gap:10px;
  justify-content:center;
  margin: 10px 0 14px 0;
}

.nav-link {
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding:8px 12px;
  border-radius:999px;
  border:1px solid var(--border);
  background: var(--pill);
  color: var(--text);
  text-decoration:none;
  font-weight:600;
  font-size:0.95rem;
  transition: transform .08s ease, border-color .08s ease, background .08s ease;
}

.nav-link:hover {
  border-color: #cbd5e1;
  background: #ffffff;
  transform: translateY(-1px);
}

.nav-link.active {
  background: rgba(37,99,235,0.10);
  border-color: rgba(37,99,235,0.35);
  color: var(--brand2);
}

.row {
  display:flex;
  gap: var(--gap);
  align-items: stretch;
  flex-wrap: wrap;
}

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 14px;
}

.card h3 { margin: 0 0 10px 0; font-size: 1.05rem; }

.controls { flex: 1 1 330px; min-width: 320px; }
.results  { flex: 2 1 560px; min-width: 420px; }

.control-label { font-size: 0.92rem; margin-top: 10px; margin-bottom: 6px; }
.small-note { color: var(--muted); font-size: 0.85rem; margin-top: 6px; }

.kpi-grid {
  display:grid;
  grid-template-columns: repeat(4, minmax(160px, 1fr));
  gap: 10px;
}

.kpi {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px 12px;
}

.kpi .kpi-title { font-size: 0.82rem; color: var(--muted); margin-bottom: 6px; }
.kpi .kpi-value { font-size: 1.25rem; font-weight: 700; }

@media (max-width: 980px){
  .kpi-grid{ grid-template-columns: repeat(2, minmax(160px, 1fr)); }
}

.graph-title { font-size: 0.95rem; font-weight: 700; margin: 0 0 8px 2px; }

.footer-note { margin-top: 16px; color: var(--muted); font-size: 0.85rem; }

hr.sep {
  border: none;
  border-top: 1px solid var(--border);
  margin: 14px 0;
}

/* Prevent the "infinite growth" feel: graphs get explicit heights in components.
   Also keep Plotly containers from inheriting odd CSS heights. */
.js-plotly-plot, .plot-container { max-height: 100%; }
        """.strip()
        + "\n",
        encoding="utf-8",
    )

# ------------------------------------------------------------
# Plotly template
# ------------------------------------------------------------
jp_template = go.layout.Template(
    layout=dict(
        margin=dict(t=60, b=60, l=70, r=40),
        title=dict(x=0.5, xanchor="center"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="lightgray"),
        yaxis=dict(showgrid=False, gridcolor="lightgray", zeroline=True, zerolinecolor="lightgray"),
    )
)
pio.templates["jp_clean"] = jp_template
pio.templates.default = "jp_clean"

_GRAPH_CONFIG = {"displayModeBar": True, "displaylogo": False, "responsive": True}

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def _fmt_usd(x: float) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "—"

def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "—"

def _safe_float(x, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def get_col_case_insensitive(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def _warn_fig(msg: str, title: str = "Not available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14),
    )
    fig.update_layout(title=title, template="jp_clean", height=360)
    return fig

# ------------------------------------------------------------
# Load main data and derive baselines
# ------------------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing core data file: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

TARGET_DEFAULT = "default"
TARGET_DEBT = "high_debt_burden"
FLAG_T1 = "can_model_default"
FLAG_T2 = "can_model_high_debt_burden"

for col in [TARGET_DEFAULT, TARGET_DEBT, FLAG_T1, FLAG_T2]:
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' not found in {DATA_PATH.name}")

df_t1 = df[df[FLAG_T1] == True].copy()
df_t2 = df[df[FLAG_T2] == True].copy()
df_both = df[(df[FLAG_T1] == True) & (df[FLAG_T2] == True)].copy()

baseline_default_rate = float(df_t1[TARGET_DEFAULT].mean()) if not df_t1.empty else float(df[TARGET_DEFAULT].mean())
baseline_high_debt_rate = float(df_t2[TARGET_DEBT].mean()) if not df_t2.empty else float(df[TARGET_DEBT].mean())

# Derive institution_type if missing and source cols exist
if ("institution_type" not in df.columns) and ("S3CLGLVL" in df.columns) and ("S3CLGCNTRL" in df.columns):
    lvl = df["S3CLGLVL"].astype("string")
    ctrl = df["S3CLGCNTRL"].astype("string")

    inst_type = pd.Series("Other/unknown", index=df.index, dtype="object")
    is_4yr = lvl.str.contains("4 or more years", case=False, na=False)
    is_2yr = lvl.str.contains("At least 2 but less than 4 years", case=False, na=False)
    is_lt2 = lvl.str.contains("Less than 2 years", case=False, na=False)

    is_public = ctrl.str.contains("Public", case=False, na=False)
    is_priv_np = ctrl.str.contains("Private not-for-profit", case=False, na=False)
    is_priv_fp = ctrl.str.contains("Private for-profit", case=False, na=False)

    inst_type[is_2yr & is_public] = "Public 2-year college"
    inst_type[is_4yr & is_public] = "Public 4-year college"
    inst_type[is_4yr & is_priv_np] = "Private non-profit 4-year college"
    inst_type[(is_2yr | is_4yr) & is_priv_fp] = "Private for-profit college"
    inst_type[is_lt2 & is_public] = "Public <2-year program"
    inst_type[is_lt2 & (is_priv_np | is_priv_fp)] = "Private <2-year program"

    df["institution_type"] = inst_type
    df_t1["institution_type"] = inst_type.loc[df_t1.index]
    df_t2["institution_type"] = inst_type.loc[df_t2.index]
    df_both["institution_type"] = inst_type.loc[df_both.index]

# Joint outcome groups for exploratory page
df_joint = df_both.copy()
df_joint["default_status"] = np.where(df_joint[TARGET_DEFAULT] == 1, "Defaulted", "No default")
df_joint["debt_status"] = np.where(df_joint[TARGET_DEBT] == 1, "High debt burden", "No high debt burden")
df_joint["default_debt_group"] = df_joint["default_status"] + " / " + df_joint["debt_status"]



# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------
# Full-length labels (for text / reporting)
variable_labels: Dict[str, str] = {
    # IDs and derived modeling targets
    "STU_ID": "Student ID (HSLS student identifier)",
    "default": "Ever defaulted on a federal student loan (binary)",
    "high_debt_burden": "Heavy debt burden (debt-to-income above threshold, binary)",
    "debt_to_income": "Debt-to-income ratio (loan balance / annual earnings)",
    "annual_earnings": "Annual earnings a few years after college (USD)",
    "earnings_source": "Source variable used to construct annual_earnings (which HSLS earnings item was used)",
    "intended_field_group": "Intended field of study (grouped)",
    "can_model_default": "Can be used for default modeling.",
    "can_model_high_debt_burden": "Can be used for high debt burden modeling.",

    # S2 earnings expectations (student)
    "S2EARN2YPUB": "Expected earnings with a two-year college degree (standardized by year)",
    "S2EARN2YPUBUN": "Expected earnings with a two-year college degree (unstandardized, original dollars)",
    "S2EARN4Y": "Expected earnings with a four-year college degree (standardized by year)",
    "S2EARN4YUN": "Expected earnings with a four-year college degree (unstandardized, original dollars)",
    "S2EARNHS": "Expected earnings with a high school diploma (standardized by year)",
    "S2EARNHSUN": "Expected earnings with a high school diploma (unstandardized, original dollars)",
    "S2EARNNOHS": "Expected earnings without a high school diploma (standardized by year)",
    "S2EARNNOHSUN": "Expected earnings without a high school diploma (unstandardized, original dollars)",
    "S2EARNOCC": "Expected earnings with an occupational training diploma/certificate (standardized by year)",
    "S2EARNOCCUN": "Expected earnings with an occupational training diploma/certificate (unstandardized, original dollars)",

    # S2 loans / financing expectations
    "S2GOVLOAN": "Plans to pay for tuition/room/board with federal or state government loans (Yes/No/Don't know)",
    "S2JOBEARN": "Current or most recent earnings since leaving high school (for dropouts/early completers)",
    "S2MAXBORROW": "Maximum amount per year student is willing to borrow to pay for education",
    "S2NODEBT": "Will not apply for financial aid or loans because does not want to go into debt",
    "S2OCC30EARN": "Expected earnings for chosen occupation at age 30",
    "S2PARPRVLOAN": "Parents have taken out private (non-federal) education loans for the student's education",
    "S2QUALGOVLOAN": "Believes will qualify for federal or state government education loans",
    "S2QUALPRVLOAN": "Believes will qualify for private (non-federal) education loans",
    "S2TEENPRVLOAN": "Student has private (non-federal) education loans (Yes/No/Don't know)",

    # S2 beliefs about learning
    "S2MLEARN": "Agreement that most people can learn to be good at math",
    "S2SLEARN": "Agreement that most people can learn to be good at science",

    # S3 FAFSA / affordability / choice set
    "S3APPFAFSA": "Completed a FAFSA for the teenager's education",
    "S3CANAFFORD": "Did not complete FAFSA because family can afford college without financial aid",
    "S3CHCCOST": "Cost of first-choice accepted school before financial aid (2013-2014 school year)",
    "S3CHCPELL": "Offered scholarship/grant to attend first-choice accepted school (2013-2014)",
    "S3CHCSTAFFORD": "Offered loan to attend first-choice accepted school (2013-2014)",

    # S3 current college (Nov 1, 2013) - finances and institution
    "S3CLGBORROW": "Total amount borrowed for college (USD)",
    "S3CLGCNTRL": "Control of enrolled college (IPEDS: public, private nonprofit, for-profit)",
    "S3CLGCOST": "Annual cost of attendance (USD)",
    "S3CLGFT": "Enrollment intensity at current college (full-time vs part-time as of Nov 1, 2013)",
    "S3CLGLVL": "Level of enrolled college (IPEDS level: 4-year, 2-year, etc.)",
    "S3CLGPELL": "Offered scholarship/grant to attend current college (Nov 1, 2013 school; 2013-2014 year)",
    "S3CLGSEL": "Selectivity of enrolled college (IPEDS selectivity code)",
    "S3CLGSTAFFORD": "Offered loan to attend current college (Nov 1, 2013 school; 2013-2014 year)",

    # S3 importance ratings & plans
    "S3COSTATTEND": "Importance of cost of attendance when choosing current college",
    "S3FIELD2": "Intended major field (2-digit CIP code) at current/considered college",
    "S3FIELD_STEM": "Indicator that intended major is in a STEM field",
    "S3FOCUS": "Main focus as of Nov 1, 2013 (e.g., attending college, working, etc.)",
    "S3GRADSCHPLC": "Importance of graduate school placement when choosing current college",
    "S3JOBPLC": "Importance of job placement when choosing current college",
    "S3NODEBT": "Did not complete FAFSA because did not want to go into debt",
    "S3OFFERSFIELD": "Importance of the particular program/field of study when choosing current college",
    "S3PROGLEVEL": "Program level of current enrollment as of Nov 1, 2013 (e.g., bachelor's, associate, certificate)",
    "S3WHERELIVE": "Where living while taking postsecondary classes (on campus, off campus, with parents, etc.)",
    "S3WORKFT": "Working full-time as of Nov 1, 2013",

    # S4 private loans
    "S4PRVLOAN": "Took out a private (non-federal) loan for college education",
    "S4PRVLOANAMT": "Total amount of private loans for college education (continuous dollar amount)",
    "S4PRVLOANEST": "Estimated total amount of private loans for college education (categorical brackets)",

    # X1 baseline school & family context
    "X1CONTROL": "School control (public, private nonprofit, private for-profit) at baseline",
    "X1FAMINCOME": "Family income (baseline, categorical; 2008 total family income from all sources)",
    "X1FREELUNCH": "Percent of 9th graders in school eligible for free/reduced-price lunch (categorical)",
    "X1HISPANIC": "Student is Hispanic/Latino/Latina (composite indicator)",
    "X1LOCALE": "School locale (urbanicity; urban, suburban, town, rural)",
    "X1PAREDEXPCT": "How far in school parent thinks the student will go",
    "X1PAREDU": "Parent education (baseline; highest level of education of parents/guardians)",
    "X1POVERTY130": "Family income at or below 130% of Census poverty threshold (indicator)",
    "X1RACE": "Student race/ethnicity (composite category)",
    "X1SCHBLACK": "Percent of students in school who are Black",
    "X1SCHHISP": "Percent of students in school who are Hispanic/Latino/Latina",
    "X1SCHWHITE": "Percent of students in school who are White",
    "X1SES": "Socioeconomic status composite (continuous index)",
    "X1SESQ5": "Socioeconomic status quintile (1=lowest, 5=highest)",
    "X1SEX": "Student sex",
    "X1STUEDEXPCT": "How far in school the 9th grader thinks he/she will get",
    "X1TXMTH": "Baseline mathematics theta score",

    # X3 postsecondary program context
    "X3PROGLEVEL": "Degree program level in 2013 Update / second follow-up (e.g., bachelor's, associate, certificate)",

    # X5 NSLDS loan amounts
    "X5OWEAMT": "Amount currently owed (principal + interest) on federal student loans as of June 30, 2016",
    "X5T4XLNCUM": "Total federal loans (excluding Parent PLUS) borrowed through June 30, 2016",

    # Additional derived variables
    "expected_salary_at_30": "Expected annual salary at age 30 (USD)",
    "institution_type": "Institution type (level and control, derived from IPEDS-level and control variables)",
}

# Short labels (for plots / axes / legends)
variable_labels_short: Dict[str, str] = {
    # IDs and derived modeling targets
    "STU_ID": "Student ID",
    "default": "Ever defaulted (fed loan)",
    "high_debt_burden": "High debt burden",
    "debt_to_income": "Debt-to-income ratio",
    "annual_earnings": "Annual earnings ($)",
    "earnings_source": "Earnings source",
    "intended_field_group": "Intended field (group)",
    "can_model_default": "Used in default model",
    "can_model_high_debt_burden": "Used in high-burden model",

    # S2 earnings expectations (student)
    "S2EARN2YPUB": "Exp earnings (2-year, std)",
    "S2EARN2YPUBUN": "Exp earnings (2-year, $)",
    "S2EARN4Y": "Exp earnings (4-year, std)",
    "S2EARN4YUN": "Exp earnings (4-year, $)",
    "S2EARNHS": "Exp earnings (HS, std)",
    "S2EARNHSUN": "Exp earnings (HS, $)",
    "S2EARNNOHS": "Exp earnings (<HS, std)",
    "S2EARNNOHSUN": "Exp earnings (<HS, $)",
    "S2EARNOCC": "Exp earnings (occ cert, std)",
    "S2EARNOCCUN": "Exp earnings (occ cert, $)",

    # S2 loans / financing expectations
    "S2GOVLOAN": "Plans gov loans",
    "S2JOBEARN": "Job earnings now ($)",
    "S2MAXBORROW": "Max willing to borrow ($/yr)",
    "S2MLEARN": "Belief can learn math",
    "S2NODEBT": "Will not borrow (avoid debt)",
    "S2OCC30EARN": "Exp earnings at 30 ($)",
    "S2PARPRVLOAN": "Parents have private loans",
    "S2QUALGOVLOAN": "Thinks qualifies gov loan",
    "S2QUALPRVLOAN": "Thinks qualifies private loan",
    "S2SLEARN": "Belief can learn science",
    "S2TEENPRVLOAN": "Has private student loans",

    # S3 FAFSA / affordability / choice set
    "S3APPFAFSA": "Completed FAFSA",
    "S3CANAFFORD": "Skip FAFSA - can afford",
    "S3CHCCOST": "Cost first-choice school ($)",
    "S3CHCPELL": "Grant/scholarship first choice",
    "S3CHCSTAFFORD": "Loan offer first choice",

    # S3 current college (Nov 1, 2013) - finances and institution
    "S3CLGBORROW": "Total borrowed for college ($)",
    "S3CLGCNTRL": "College control",
    "S3CLGCOST": "Cost current college ($)",
    "S3CLGFT": "Full-time enrollment",
    "S3CLGLVL": "College level",
    "S3CLGPELL": "Grant/scholarship current college",
    "S3CLGSEL": "College selectivity",
    "S3CLGSTAFFORD": "Loan offer current college",

    # S3 importance ratings & plans
    "S3COSTATTEND": "Importance - cost",
    "S3FIELD2": "Intended major (2-digit)",
    "S3FIELD_STEM": "Major is STEM",
    "S3FOCUS": "Main focus/activity",
    "S3GRADSCHPLC": "Importance - grad placement",
    "S3JOBPLC": "Importance - job placement",
    "S3NODEBT": "Skip FAFSA - avoid debt",
    "S3OFFERSFIELD": "Importance - has program/field",
    "S3PROGLEVEL": "Program level (current)",
    "S3WHERELIVE": "Where student lives",
    "S3WORKFT": "Works full-time",

    # S4 private loans
    "S4PRVLOAN": "Has private loans",
    "S4PRVLOANAMT": "Total private loans ($)",
    "S4PRVLOANEST": "Total private loans (bracket)",

    # X1 baseline school & family context
    "X1CONTROL": "HS control",
    "X1FAMINCOME": "Family income (baseline)",
    "X1FREELUNCH": "% eligible free lunch",
    "X1HISPANIC": "Hispanic",
    "X1LOCALE": "School locale",
    "X1PAREDEXPCT": "Parent educ expectation",
    "X1PAREDU": "Parent education",
    "X1POVERTY130": "Income <=130% poverty",
    "X1RACE": "Race/ethnicity",
    "X1SCHBLACK": "% Black in school",
    "X1SCHHISP": "% Hispanic in school",
    "X1SCHWHITE": "% White in school",
    "X1SES": "SES index",
    "X1SESQ5": "SES quintile (1-5)",
    "X1SEX": "Sex",
    "X1STUEDEXPCT": "Student educ expectation",
    "X1TXMTH": "Baseline math score",

    # X3 postsecondary program context
    "X3PROGLEVEL": "Program level (2013)",

    # X5 NSLDS loan amounts
    "X5OWEAMT": "Amount owed fed loans ($)",
    "X5T4XLNCUM": "Total fed loans ($)",

    # Additional derived variables
    "expected_salary_at_30": "Expected salary at 30 ($)",
    "institution_type": "Institution type",
}

# Explanations for one-hot-encoded categorical dummies (full and short)
categorical_var_explanations_short: Dict[str, str] = {
    # Institution type dummies (from institution_type)
    "institution_type_Private <2-year program": "Private <2-year program",
    "institution_type_Private for-profit college": "Private for-profit college",
    "institution_type_Other/unknown": "Other/unknown institution",
    "institution_type_Public <2-year program": "Public <2-year program",
    "institution_type_Private non-profit 4-year college": "Private non-profit 4-year",
    "institution_type_Public 2-year college": "Public 2-year college",
    "institution_type_Public 4-year college": "Public 4-year college",

    # Family income brackets (from X1FAMINCOME)
    "X1FAMINCOME_Family income > $235,000": "Family income > $235k",
    "X1FAMINCOME_Family income less than or equal to $15,000": "Family income ≤ $15k",
    "X1FAMINCOME_Family income > $175,000 and <= $195,000": "Family income $175k-$195k",
    "X1FAMINCOME_Family income > $195,000 and <= $215,000": "Family income $195k-$215k",
    "X1FAMINCOME_Family income > $15,000 and <= $35,000": "Family income $15k-$35k",
    "X1FAMINCOME_Family income > $215,000 and <= $235,000": "Family income $215k-$235k",
    "X1FAMINCOME_Family income > $35,000 and <= $55,000": "Family income $35k-$55k",
    "X1FAMINCOME_Not reported": "Family income not reported",
    "X1FAMINCOME_Family income > $135,000 and <= $155,000": "Family income $135k-$155k",
    "X1FAMINCOME_Family income > $75,000 and <= $95,000": "Family income $75k-$95k",
    "X1FAMINCOME_Family income > $55,000 and <= $75,000": "Family income $55k-$75k",
    "X1FAMINCOME_Family income > $115,000 and <= $135,000": "Family income $115k-$135k",
    "X1FAMINCOME_Family income > $155,000 and <=$175,000": "Family income $155k-$175k",
    "X1FAMINCOME_Family income > $95,000 and <= $115,000": "Family income $95k-$115k",

    # Parent education dummies (from X1PAREDU)
    "X1PAREDU_Less than high school": "Parent ed < HS",
    "X1PAREDU_Not reported": "Parent ed not reported",
    "X1PAREDU_Associate's degree": "Parent ed AA",
    "X1PAREDU_Master's degree": "Parent ed MA",
    "X1PAREDU_Ph.D/M.D/Law/other high lvl prof degree": "Parent ed PhD/MD/JD+",
    "X1PAREDU_Bachelor's degree": "Parent ed BA/BS",
    "X1PAREDU_High school diploma or GED": "Parent ed HS/GED",

    # SES quintile dummies (from X1SESQ5)
    "X1SESQ5_Not reported": "SES quintile not reported",
    "X1SESQ5_Fifth quintile (highest)": "SES Q5 (highest)",
    "X1SESQ5_Second quintile": "SES Q2",
    "X1SESQ5_Fourth quintile": "SES Q4",
    "X1SESQ5_Third quintile": "SES Q3",
    "X1SESQ5_First quintile (lowest)": "SES Q1 (lowest)",

    # Intended field-of-study group dummies (from intended_field_group)
    "intended_field_group_Humanities/other": "Humanities/other",
    "intended_field_group_Education": "Education",
    "intended_field_group_Undecided": "Undecided",
    "intended_field_group_Social sciences": "Social sciences",
    "intended_field_group_Health": "Health fields",
    "intended_field_group_Business": "Business",
    "intended_field_group_Trades": "Trades/vocational",
    "intended_field_group_Liberal arts": "Liberal arts",
    "intended_field_group_STEM": "STEM",

    # Original continuous variables referenced directly
    "S3CLGCOST": "College cost ($/yr)",
    "S3CLGBORROW": "Total borrowed ($)",
    "expected_salary_at_30": "Expected salary at 30 ($)",}

def pretty_label(var_name: str) -> str:
    return variable_labels.get(var_name, var_name)

def short_label(var_name: str, max_len: int = 60) -> str:
    lbl = variable_labels_short.get(var_name, pretty_label(var_name))
    if "(" in lbl:
        lbl = lbl.split("(", 1)[0].strip()
    if len(lbl) > max_len:
        lbl = lbl[: max_len - 3] + "..."
    return lbl

def make_feature_readable(feature_name: str) -> str:
    if feature_name in categorical_var_explanations_short:
        return categorical_var_explanations_short[feature_name]
    if "_Missing" in feature_name:
        base, _ = feature_name.split("_Missing", 1)
        base_label = variable_labels_short.get(base, base)
        return f"{base_label} missing/not reported"
    if feature_name in variable_labels_short:
        return variable_labels_short[feature_name]
    if "_" in feature_name:
        base, category = feature_name.split("_", 1)
        base_label = variable_labels_short.get(base, base)
        return f"{base_label}: {category}"
    return feature_name

# build short-label order
income_short_order = [
    categorical_var_explanations_short["X1FAMINCOME_Family income less than or equal to $15,000"],
    categorical_var_explanations_short["X1FAMINCOME_Family income > $15,000 and <= $35,000"],
    categorical_var_explanations_short["X1FAMINCOME_Family income > $35,000 and <= $55,000"],
    categorical_var_explanations_short["X1FAMINCOME_Family income > $55,000 and <= $75,000"],
    categorical_var_explanations_short["X1FAMINCOME_Family income > $75,000 and <= $95,000"],
    categorical_var_explanations_short["X1FAMINCOME_Family income > $95,000 and <= $115,000"],
    categorical_var_explanations_short["X1FAMINCOME_Family income > $115,000 and <= $135,000"],
    categorical_var_explanations_short["X1FAMINCOME_Family income > $135,000 and <= $155,000"],
    categorical_var_explanations_short["X1FAMINCOME_Family income > $155,000 and <=$175,000"],
    categorical_var_explanations_short["X1FAMINCOME_Family income > $175,000 and <= $195,000"],
    categorical_var_explanations_short["X1FAMINCOME_Family income > $195,000 and <= $215,000"],
    categorical_var_explanations_short["X1FAMINCOME_Family income > $215,000 and <= $235,000"],
    categorical_var_explanations_short["X1FAMINCOME_Family income > $235,000"],
    categorical_var_explanations_short["X1FAMINCOME_Not reported"],
]

# create plotting column with those short labels
fi_raw = df_joint["X1FAMINCOME"].astype("string").fillna("Missing")
fi_raw_l = fi_raw.str.lower()

df_joint["X1FAMINCOME_plot"] = pd.Series(index=df_joint.index, dtype="object")

patterns = [
    (r"less than or equal to\s*\$15,000",
     categorical_var_explanations_short["X1FAMINCOME_Family income less than or equal to $15,000"]),
    (r">\s*\$15,000\s*and\s*<=\s*\$35,000",
     categorical_var_explanations_short["X1FAMINCOME_Family income > $15,000 and <= $35,000"]),
    (r">\s*\$35,000\s*and\s*<=\s*\$55,000",
     categorical_var_explanations_short["X1FAMINCOME_Family income > $35,000 and <= $55,000"]),
    (r">\s*\$55,000\s*and\s*<=\s*\$75,000",
     categorical_var_explanations_short["X1FAMINCOME_Family income > $55,000 and <= $75,000"]),
    (r">\s*\$75,000\s*and\s*<=\s*\$95,000",
     categorical_var_explanations_short["X1FAMINCOME_Family income > $75,000 and <= $95,000"]),
    (r">\s*\$95,000\s*and\s*<=\s*\$115,000",
     categorical_var_explanations_short["X1FAMINCOME_Family income > $95,000 and <= $115,000"]),
    (r">\s*\$115,000\s*and\s*<=\s*\$135,000",
     categorical_var_explanations_short["X1FAMINCOME_Family income > $115,000 and <= $135,000"]),
    (r">\s*\$135,000\s*and\s*<=\s*\$155,000",
     categorical_var_explanations_short["X1FAMINCOME_Family income > $135,000 and <= $155,000"]),
    (r">\s*\$155,000\s*and\s*<=\s*\$175,000",
     categorical_var_explanations_short["X1FAMINCOME_Family income > $155,000 and <=$175,000"]),
    (r">\s*\$175,000\s*and\s*<=\s*\$195,000",
     categorical_var_explanations_short["X1FAMINCOME_Family income > $175,000 and <= $195,000"]),
    (r">\s*\$195,000\s*and\s*<=\s*\$215,000",
     categorical_var_explanations_short["X1FAMINCOME_Family income > $195,000 and <= $215,000"]),
    (r">\s*\$215,000\s*and\s*<=\s*\$235,000",
     categorical_var_explanations_short["X1FAMINCOME_Family income > $215,000 and <= $235,000"]),
    (r"family income\s*>\s*\$235,000",
     categorical_var_explanations_short["X1FAMINCOME_Family income > $235,000"]),
    (r"not reported",
     categorical_var_explanations_short["X1FAMINCOME_Not reported"]),
]

for pat, lbl in patterns:
    m = fi_raw_l.str.contains(pat, regex=True, na=False)
    df_joint.loc[m, "X1FAMINCOME_plot"] = lbl

df_joint.loc[df_joint["X1FAMINCOME_plot"].isna(), "X1FAMINCOME_plot"] = "Family income: other/unknown"
if "Family income: other/unknown" in df_joint["X1FAMINCOME_plot"].unique():
    income_short_order = income_short_order + ["Family income: other/unknown"]

df_joint["X1FAMINCOME_plot"] = pd.Categorical(
    df_joint["X1FAMINCOME_plot"],
    categories=income_short_order,
    ordered=True,
)

# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------
model_default_app = None
model_high_debt_app = None
model_load_warnings: List[str] = []

try:
    model_default_app = joblib.load(MODEL_DEFAULT_APP_PATH)
except Exception as e:
    model_load_warnings.append(f"Could not load default model: {e}")

try:
    model_high_debt_app = joblib.load(MODEL_HIGHDEBT_APP_PATH)
except Exception as e:
    model_load_warnings.append(f"Could not load high-debt model: {e}")

# ------------------------------------------------------------
# App-ready feature set and UI defaults
# ------------------------------------------------------------
APP_FEATURES = [
    "S3CLGBORROW",
    "S3CLGCOST",
    "intended_field_group",
    "institution_type",
    "X1FAMINCOME",
    "X1PAREDU",
    "X1SEX",
    "S3CLGSEL",
]

for feat in APP_FEATURES:
    if feat not in df.columns:
        raise ValueError(f"Expected app feature '{feat}' not found in the data. Present columns differ from this app.")

numeric_app = [c for c in APP_FEATURES if is_numeric_dtype(df[c])]
categorical_app = [c for c in APP_FEATURES if c not in numeric_app]

default_values: Dict[str, object] = {}
for col in numeric_app:
    default_values[col] = float(pd.to_numeric(df[col], errors="coerce").median())
for col in categorical_app:
    vc = df[col].value_counts(dropna=True)
    default_values[col] = vc.index[0] if not vc.empty else None

default_values.update(
    {
        "S3CLGBORROW": 0.0,
        "S3CLGCOST": 21000.0,
        "intended_field_group": "Undecided",
        "institution_type": "Other/unknown",
        "X1FAMINCOME": "Not reported",
        "X1PAREDU": "Not reported",
        "X1SEX": "Not reported",
        "S3CLGSEL": "Not reported",
    }
)

def slider_range(col: str, lower_q: float = 0.01, upper_q: float = 0.99) -> Tuple[float, float]:
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return 0.0, 1.0
    lo = float(series.quantile(lower_q))
    hi = float(series.quantile(upper_q))
    if lo < 0:
        lo = 0.0
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi

def money_label(value: float) -> str:
    value = float(value)
    if value >= 1000:
        return f"${int(round(value / 1000))}k"
    return f"${int(round(value))}"

borrow_min, borrow_max = slider_range("S3CLGBORROW")
cost_min, cost_max = slider_range("S3CLGCOST")

borrow_marks = {float(borrow_min): money_label(borrow_min), float(borrow_max): money_label(borrow_max)}
cost_marks_initial = {float(cost_min): money_label(cost_min), float(cost_max): money_label(cost_max)}

def dropdown_options(col: str) -> List[dict]:
    values = df[col].dropna().unique().tolist()
    try:
        values = sorted(values)
    except Exception:
        pass
    return [{"label": str(v), "value": v} for v in values]

# ------------------------------------------------------------
# Risk helpers + prediction wrapper
# ------------------------------------------------------------
def risk_band_default(prob: float) -> str:
    if prob < 0.015:
        return "Very low"
    elif prob < 0.035:
        return "Low"
    elif prob < 0.08:
        return "Moderate"
    elif prob < 0.14:
        return "High"
    else:
        return "Very high"

def risk_band_debt(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    elif prob < 0.50:
        return "Moderate"
    elif prob < 0.65:
        return "High"
    else:
        return "Very high"
    
# Gauge band definitions (must match risk_band_* thresholds exactly)
DEFAULT_RISK_BANDS = [
    ("Very low", 0.00, 0.015),
    ("Low",      0.015, 0.035),
    ("Moderate", 0.035, 0.08),
    ("High",     0.08,  0.14),
    ("Very high",0.14,  1.00),
]
DEBT_RISK_BANDS = [
    ("Low",      0.00, 0.30),
    ("Moderate", 0.30, 0.50),
    ("High",     0.50, 0.65),
    ("Very high",0.65, 1.00),
]

# Green -> red (one color per band, same size blocks)
DEFAULT_RISK_COLORS = ["#16a34a", "#84cc16", "#facc15", "#fb923c", "#ef4444"]
DEBT_RISK_COLORS    = ["#16a34a", "#facc15", "#fb923c", "#ef4444"]

def _prob_to_equal_band_axis(p: float, bands: List[Tuple[str, float, float]]) -> float:
    """
    Map a probability p in [0,1] into an axis where each band has equal width.
    Axis range becomes [0, n_bands]. Within-band position is linear.
    """
    p = float(np.clip(p, 0.0, 1.0))
    for i, (_, lo, hi) in enumerate(bands):
        is_last = (i == len(bands) - 1)
        if (p < hi) or is_last:
            frac = 0.0 if hi <= lo else (p - lo) / (hi - lo)
            frac = float(np.clip(frac, 0.0, 1.0))
            return i + frac
    return float(len(bands))


def _percentile(scores: np.ndarray, p: float) -> Optional[float]:
    if scores is None or scores.size == 0 or np.isnan(p):
        return None
    return float(np.mean(scores <= p))

def predict_student_risks_app(new_students_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in APP_FEATURES if c not in new_students_df.columns]
    if missing:
        raise ValueError(f"new_students_df is missing required app features: {missing}")

    if model_default_app is None or model_high_debt_app is None:
        out = pd.DataFrame(index=new_students_df.index)
        out["prob_default"] = np.nan
        out["prob_high_debt"] = np.nan
        out["default_risk_vs_average"] = np.nan
        out["high_debt_risk_vs_average"] = np.nan
        out["default_risk_band"] = ""
        out["high_debt_risk_band"] = ""
        return out

    X_new = new_students_df[APP_FEATURES].copy()

    p_default = model_default_app.predict_proba(X_new)[:, 1]
    p_debt = model_high_debt_app.predict_proba(X_new)[:, 1]

    rel_default = p_default / max(baseline_default_rate, 1e-12)
    rel_debt = p_debt / max(baseline_high_debt_rate, 1e-12)

    return pd.DataFrame(
        {
            "prob_default": p_default,
            "default_risk_band": [risk_band_default(p) for p in p_default],
            "default_risk_vs_average": rel_default,
            "prob_high_debt": p_debt,
            "high_debt_risk_band": [risk_band_debt(p) for p in p_debt],
            "high_debt_risk_vs_average": rel_debt,
        },
        index=new_students_df.index,
    )

# ------------------------------------------------------------
# Static logistic demo: borrowing vs probability of default
# ------------------------------------------------------------
logistic_demo_fig = None
try:
    mask = df_t1["S3CLGBORROW"].notna() & df_t1[TARGET_DEFAULT].notna()
    df_balance = df_t1.loc[mask, ["S3CLGBORROW", TARGET_DEFAULT]].copy()

    if len(df_balance) >= 50:
        X_balance = df_balance[["S3CLGBORROW"]].to_numpy()
        y_balance = df_balance[TARGET_DEFAULT].to_numpy()

        logit_balance = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000)
        logit_balance.fit(X_balance, y_balance)

        balance_grid = np.linspace(max(0, borrow_min), borrow_max, 200).reshape(-1, 1)
        prob_grid = logit_balance.predict_proba(balance_grid)[:, 1]

        sample_n = min(2000, len(df_balance))
        rng = np.random.default_rng(242)
        scatter_df = df_balance.sample(n=sample_n, random_state=242).copy()
        jitter = rng.uniform(-0.03, 0.03, size=sample_n)
        scatter_df["default_jitter"] = (scatter_df[TARGET_DEFAULT] + jitter).clip(-0.05, 1.05)

        logistic_demo_fig = go.Figure()
        logistic_demo_fig.add_trace(
            go.Scatter(
                x=scatter_df["S3CLGBORROW"],
                y=scatter_df["default_jitter"],
                mode="markers",
                name="Students",
                marker=dict(size=5, opacity=0.4),
            )
        )
        logistic_demo_fig.add_trace(
            go.Scatter(x=balance_grid.ravel(), y=prob_grid, mode="lines", name="Fitted logistic curve")
        )
        logistic_demo_fig.update_layout(
            title="Toy demo: probability of default vs total borrowed (single-variable logistic fit)",
            xaxis_title=short_label("S3CLGBORROW"),
            yaxis_title="Probability of default",
            template="jp_clean",
            height=420,
        )
        logistic_demo_fig.update_yaxes(range=[-0.05, 1.05])
    else:
        logistic_demo_fig = _warn_fig("Not enough data to fit the toy logistic demo.", "Toy logistic demo")
except Exception as e:
    logistic_demo_fig = _warn_fig(f"Could not build toy logistic demo: {e}", "Toy logistic demo")

# ------------------------------------------------------------
# Load CSVs
# ------------------------------------------------------------
model_perf_df = _read_csv(MODEL_PERF_PATH)

coef_default_df = _read_csv(COEF_DEFAULT_APP_PATH)
coef_highdebt_df = _read_csv(COEF_HIGHDEBT_APP_PATH)

perm_default_df = _read_csv(PERM_DEFAULT_APP_PATH)
perm_highdebt_df = _read_csv(PERM_HIGHDEBT_APP_PATH)

cal_default_app_df = _read_csv(CAL_DEFAULT_APP_PATH)
cal_highdebt_app_df = _read_csv(CAL_HIGHDEBT_APP_PATH)
cal_default_full_df = _read_csv(CAL_DEFAULT_FULL_PATH)
cal_highdebt_full_df = _read_csv(CAL_HIGHDEBT_FULL_PATH)

roc_default_app_df = _read_csv(ROC_DEFAULT_APP_PATH)
roc_highdebt_app_df = _read_csv(ROC_HIGHDEBT_APP_PATH)
roc_default_full_df = _read_csv(ROC_DEFAULT_FULL_PATH)
roc_highdebt_full_df = _read_csv(ROC_HIGHDEBT_FULL_PATH)

pr_default_app_df = _read_csv(PR_DEFAULT_APP_PATH)
pr_highdebt_app_df = _read_csv(PR_HIGHDEBT_APP_PATH)
pr_default_full_df = _read_csv(PR_DEFAULT_FULL_PATH)
pr_highdebt_full_df = _read_csv(PR_HIGHDEBT_FULL_PATH)

sweep_default_app_df = _read_csv(SWEEP_DEFAULT_APP_PATH)
sweep_highdebt_app_df = _read_csv(SWEEP_HIGHDEBT_APP_PATH)
sweep_default_full_df = _read_csv(SWEEP_DEFAULT_FULL_PATH)
sweep_highdebt_full_df = _read_csv(SWEEP_HIGHDEBT_FULL_PATH)

thr_tune_default_full_df = _read_csv(THR_TUNE_DEFAULT_FULL_PATH)
thr_tune_highdebt_full_df = _read_csv(THR_TUNE_HIGHDEBT_FULL_PATH)

pca_df = _read_csv(PCA_EMBEDDINGS_PATH)
pca_var_df = _read_csv(PCA_VAR_PATH)
cluster_summary_df = _read_csv(CLUSTER_SUMMARY_PATH)

# OOF predictions for percentile ranks
oof_default_app = _read_csv(OOF_DEFAULT_APP_PATH)
oof_highdebt_app = _read_csv(OOF_HIGHDEBT_APP_PATH)
oof_default_scores = (
    pd.to_numeric(oof_default_app["y_score"], errors="coerce").dropna().to_numpy()
    if "y_score" in oof_default_app.columns
    else np.array([])
)
oof_highdebt_scores = (
    pd.to_numeric(oof_highdebt_app["y_score"], errors="coerce").dropna().to_numpy()
    if "y_score" in oof_highdebt_app.columns
    else np.array([])
)

def prepare_coef_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_local = df_raw.copy()
    lower_map = {c.lower(): c for c in df_local.columns}

    feat_col = None
    for cand in ["feature_readable", "feature", "variable", "var", "name"]:
        if cand in lower_map:
            feat_col = lower_map[cand]
            break

    coef_col = None
    for cand in ["coefficient", "coef", "logit_coef", "log_odds", "estimate"]:
        if cand in lower_map:
            coef_col = lower_map[cand]
            break

    if feat_col is None or coef_col is None:
        return pd.DataFrame()

    df_out = df_local[[feat_col, coef_col]].rename(columns={feat_col: "Feature", coef_col: "Coefficient"})
    return df_out

def prepare_perm_importance_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_local = df_raw.copy()
    lower_map = {c.lower(): c for c in df_local.columns}

    feat_col = None
    for cand in ["feature", "variable", "var", "name"]:
        if cand in lower_map:
            feat_col = lower_map[cand]
            break

    imp_col = None
    for cand in ["importance", "perm_importance", "mean_importance", "imp"]:
        if cand in lower_map:
            imp_col = lower_map[cand]
            break

    if feat_col is None or imp_col is None:
        return pd.DataFrame()

    df_out = df_local[[feat_col, imp_col]].rename(columns={feat_col: "feature", imp_col: "importance"})
    df_out["feature_readable"] = df_out["feature"].apply(make_feature_readable)
    return df_out

coef_default = prepare_coef_df(coef_default_df)
coef_debt = prepare_coef_df(coef_highdebt_df)
perm_import_default = prepare_perm_importance_df(perm_default_df)
perm_import_debt = prepare_perm_importance_df(perm_highdebt_df)

# ------------------------------------------------------------
# Figures for diagnostics pages
# ------------------------------------------------------------
def build_calibration_fig(df_raw: pd.DataFrame, title: str) -> go.Figure:
    if df_raw is None or df_raw.empty:
        return _warn_fig("Calibration curve CSV not available.", title)

    df_local = df_raw.copy()
    lower_map = {c.lower(): c for c in df_local.columns}
    pred_col = get_col_case_insensitive(df_local, ["predicted", "mean_predicted", "pred_mid", "pred", "y_score"])
    obs_col = get_col_case_insensitive(df_local, ["observed", "empirical", "true", "rate", "y_true"])

    if pred_col is None or obs_col is None:
        if len(df_local.columns) >= 2:
            pred_col, obs_col = df_local.columns[:2]
        else:
            return _warn_fig("Could not infer predicted/observed columns.", title)

    fig = px.line(
        df_local,
        x=pred_col,
        y=obs_col,
        markers=True,
        labels={pred_col: "Mean predicted probability", obs_col: "Observed frequency"},
        title=title,
        template="jp_clean",
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(height=420)
    return fig

def _concat_with_model(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    dfs = [d for d in df_list if d is not None and not d.empty]
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    return out


def build_roc_fig(df_app: pd.DataFrame, df_full: pd.DataFrame, title: str) -> go.Figure:
    df_all = _concat_with_model([df_app, df_full])
    if df_all.empty:
        return _warn_fig("ROC CSVs not available.", title)

    # Identify columns
    fpr = get_col_case_insensitive(df_all, ["fpr", "false_positive_rate", "x"])
    tpr = get_col_case_insensitive(df_all, ["tpr", "true_positive_rate", "y"])
    if fpr is None or tpr is None:
        # Fallback: assume first two numeric columns are x/y
        cols = df_all.columns.tolist()
        if len(cols) < 2:
            return _warn_fig("ROC CSVs missing usable columns.", title)
        fpr, tpr = cols[:2]

    if "model" not in df_all.columns:
        # Backward-compatible fallback
        df_all["model"] = "Model"

    fig = go.Figure()

    # One trace per model (auto-colored by Plotly)
    for m, d in df_all.groupby("model", dropna=False):
        d = d.sort_values(fpr)
        fig.add_trace(
            go.Scatter(
                x=d[fpr],
                y=d[tpr],
                mode="lines",
                name=str(m),
            )
        )

    # Chance line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        template="jp_clean",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def build_pr_fig(df_app: pd.DataFrame, df_full: pd.DataFrame, title: str) -> go.Figure:
    df_all = _concat_with_model([df_app, df_full])
    if df_all.empty:
        return _warn_fig("PR CSVs not available.", title)

    # Identify columns
    prec = get_col_case_insensitive(df_all, ["precision", "prec", "y"])
    rec = get_col_case_insensitive(df_all, ["recall", "tpr", "x"])
    if rec is None or prec is None:
        cols = df_all.columns.tolist()
        if len(cols) < 2:
            return _warn_fig("PR CSVs missing usable columns.", title)
        rec, prec = cols[:2]

    if "model" not in df_all.columns:
        df_all["model"] = "Model"

    fig = go.Figure()

    for m, d in df_all.groupby("model", dropna=False):
        d = d.sort_values(rec)
        fig.add_trace(
            go.Scatter(
                x=d[rec],
                y=d[prec],
                mode="lines",
                name=str(m),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="jp_clean",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def build_sweep_fig(df_app: pd.DataFrame, df_full: pd.DataFrame, title: str) -> go.Figure:
    df_all = _concat_with_model([df_app, df_full])
    if df_all.empty:
        return _warn_fig("Threshold sweep CSVs not available.", title)

    thr = get_col_case_insensitive(df_all, ["threshold", "thr", "cutoff", "x"])
    metric = get_col_case_insensitive(df_all, ["f1", "balanced_accuracy", "youden_j", "metric", "y"])
    if thr is None or metric is None:
        cols = df_all.columns.tolist()
        if len(cols) < 2:
            return _warn_fig("Sweep CSVs missing usable columns.", title)
        thr, metric = cols[:2]

    if "model" not in df_all.columns:
        df_all["model"] = "Model"

    fig = go.Figure()

    for m, d in df_all.groupby("model", dropna=False):
        d = d.sort_values(thr)
        fig.add_trace(
            go.Scatter(
                x=d[thr],
                y=d[metric],
                mode="lines",
                name=str(m),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Threshold",
        yaxis_title="F1 Metric",
        template="jp_clean",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


# ------------------------------------------------------------
# PCA figure helper
# ------------------------------------------------------------
def fig_pca_scatter(color_by: str) -> go.Figure:
    if pca_df.empty or "PC1" not in pca_df.columns or "PC2" not in pca_df.columns:
        return _warn_fig("PCA embeddings not available.", "Borrower profiles (PCA)")

    dfp = pca_df.copy()
    if "cluster" in dfp.columns:
        dfp["cluster"] = dfp["cluster"].astype(str)

    if color_by not in dfp.columns:
        color_by = "cluster" if "cluster" in dfp.columns else dfp.columns[0]

    if color_by in {TARGET_DEFAULT, TARGET_DEBT}:
        dfp[color_by] = pd.to_numeric(dfp[color_by], errors="coerce").fillna(0).astype(int).astype(str)

    fig = px.scatter(
        dfp,
        x="PC1",
        y="PC2",
        color=color_by,
        title=f"Borrower profiles (PCA) colored by {color_by}",
        labels={"PC1": "Principal component 1", "PC2": "Principal component 2", color_by: color_by},
        template="jp_clean",
        hover_data=[c for c in ["cluster", TARGET_DEFAULT, TARGET_DEBT] if c in dfp.columns],
    )
    fig.update_layout(height=560)
    return fig

def fig_pca_variance() -> go.Figure:
    if pca_var_df.empty:
        return _warn_fig("Explained variance CSV not available.", "PCA explained variance")

    dfv = pca_var_df.copy()

    comp_col = get_col_case_insensitive(dfv, ["component", "pc", "name"])
    evr_col = get_col_case_insensitive(dfv, ["explained_variance_ratio", "evr", "variance_explained", "ratio"])

    if comp_col and evr_col:
        plot_df = dfv[[comp_col, evr_col]].copy()
        plot_df[comp_col] = plot_df[comp_col].astype(str)
        fig = px.bar(
            plot_df,
            x=comp_col,
            y=evr_col,
            title="Explained variance by principal component",
            labels={comp_col: "Component", evr_col: "Explained variance ratio"},
            template="jp_clean",
        )
        fig.update_layout(height=420)
        return fig

    numeric_cols = [c for c in dfv.columns if re.match(r"(?i)^pc\d+$", str(c).strip())]
    if numeric_cols:
        vals = pd.to_numeric(dfv.loc[dfv.index[0], numeric_cols], errors="coerce").fillna(0).to_numpy()
        plot_df = pd.DataFrame({"Component": numeric_cols, "Explained variance ratio": vals})
        fig = px.bar(
            plot_df,
            x="Component",
            y="Explained variance ratio",
            title="Explained variance by principal component",
            template="jp_clean",
        )
        fig.update_layout(height=420)
        return fig

    return _warn_fig("Could not infer explained-variance format.", "PCA explained variance")

# ------------------------------------------------------------
# Ordering helper used in exploratory plots
# ------------------------------------------------------------
def _order_fam_income_categories(series: pd.Series) -> List:
    vals = series.dropna().unique().tolist()

    def fam_key(v):
        txt = str(v).lower()
        if "not reported" in txt:
            return float("inf")
        nums = re.findall(r"\$([0-9,]+)", str(v))
        if nums:
            nums_clean = [int(n.replace(",", "")) for n in nums]
            return min(nums_clean)
        return float("inf") - 1

    return sorted(vals, key=fam_key)

# ------------------------------------------------------------
# Page layouts
# ------------------------------------------------------------
def _kpi(title: str, value: str) -> html.Div:
    return html.Div(
        className="kpi",
        children=[
            html.Div(title, className="kpi-title"),
            html.Div(value, className="kpi-value"),
        ],
    )

def layout_header(pathname: str) -> html.Div:
    def link(label: str, href: str) -> dcc.Link:
        cls = "nav-link active" if pathname == href else "nav-link"
        return dcc.Link(label, href=href, className=cls)

    warning_block = []
    if model_load_warnings:
        warning_block = [
            html.Div(
                className="card",
                children=[
                    html.H3("Startup warnings"),
                    html.Ul([html.Li(w) for w in model_load_warnings]),
                    html.Div(
                        "The app still runs, but prediction outputs will be blank until models load correctly.",
                        className="small-note",
                    ),
                ],
            )
        ]

    return html.Div(
        className="header",
        children=[
            html.H1("Student Loan Risk Explorer", style={"margin": "6px 0 2px 0"}),
            html.Div(
                [
                    html.Span("Baseline default rate: ", className="subtle"),
                    html.Span(_fmt_pct(baseline_default_rate), className="subtle"),
                    html.Span(" • ", className="subtle"),
                    html.Span("Baseline high-debt rate: ", className="subtle"),
                    html.Span(_fmt_pct(baseline_high_debt_rate), className="subtle"),
                ]
            ),
            html.Div(
                className="navbar",
                children=[
                    link("Risk Explorer", "/"),
                    link("Model Diagnostics", "/models"),
                    link("PCA Profiles", "/pca"),
                    link("Exploratory Patterns", "/exploratory"),
                    link("Feature Effects", "/features"),
                    link("Data", "/data"),
                ],
            ),
            *warning_block,
        ],
    )

def layout_page_risk_explorer() -> html.Div:
    return html.Div(
        [
            html.Div(
                className="row",
                children=[
                    # Controls
                    html.Div(
                        className="card controls",
                        children=[
                            html.H3("Student profile and college choices"),
                            html.Div(short_label("S3CLGBORROW"), className="control-label"),
                            dcc.Slider(
                                id="input_S3CLGBORROW",
                                min=borrow_min,
                                max=borrow_max,
                                step=1000,
                                value=default_values["S3CLGBORROW"],
                                tooltip={"placement": "bottom", "always_visible": True},
                                marks=borrow_marks,
                            ),
                            html.Div(short_label("S3CLGCOST"), className="control-label"),
                            dcc.Slider(
                                id="input_S3CLGCOST",
                                min=cost_min,
                                max=cost_max,
                                step=1000,
                                value=default_values["S3CLGCOST"],
                                tooltip={"placement": "bottom", "always_visible": True},
                                marks=cost_marks_initial,
                            ),
                            html.Div(short_label("intended_field_group"), className="control-label"),
                            dcc.Dropdown(
                                id="input_intended_field_group",
                                options=dropdown_options("intended_field_group"),
                                value=default_values["intended_field_group"],
                                clearable=False,
                            ),
                            html.Div(short_label("institution_type"), className="control-label"),
                            dcc.Dropdown(
                                id="input_institution_type",
                                options=dropdown_options("institution_type"),
                                value=default_values["institution_type"],
                                clearable=False,
                            ),
                            html.Div(short_label("X1FAMINCOME"), className="control-label"),
                            dcc.Dropdown(
                                id="input_X1FAMINCOME",
                                options=dropdown_options("X1FAMINCOME"),
                                value=default_values["X1FAMINCOME"],
                                clearable=False,
                            ),
                            html.Div(short_label("X1PAREDU"), className="control-label"),
                            dcc.Dropdown(
                                id="input_X1PAREDU",
                                options=dropdown_options("X1PAREDU"),
                                value=default_values["X1PAREDU"],
                                clearable=False,
                            ),
                            html.Div(short_label("X1SEX"), className="control-label"),
                            dcc.Dropdown(
                                id="input_X1SEX",
                                options=dropdown_options("X1SEX"),
                                value=default_values["X1SEX"],
                                clearable=False,
                            ),
                            html.Div(short_label("S3CLGSEL"), className="control-label"),
                            dcc.Dropdown(
                                id="input_S3CLGSEL",
                                options=dropdown_options("S3CLGSEL"),
                                value=default_values["S3CLGSEL"],
                                clearable=False,
                            ),
                            html.Div(
                                "Sliders use 1st-99th percentile ranges from the HSLS modeling sample to avoid extreme outliers.",
                                className="small-note",
                            ),
                        ],
                    ),

                    # Results
                    html.Div(
                        className="card results",
                        children=[
                            html.H3("Predicted risk for this student"),
                            html.Div(id="kpi_row", className="kpi-grid"),
                            html.Hr(className="sep"),

                            html.Div(
                                className="row",
                                children=[
                                    html.Div(
                                        style={"flex": "1 1 360px", "minWidth": "320px"},
                                        children=[
                                            html.Div("Default risk level", className="graph-title"),
                                            html.Div(
                                                style={
                                                    "width": "min(100%, 460px)",
                                                    "minWidth": "320px",
                                                    "aspectRatio": "16 / 9",
                                                    "position": "relative",
                                                    "margin": "0 auto",
                                                },
                                                children=[
                                                    dcc.Graph(
                                                        id="gauge_default",
                                                        config=_GRAPH_CONFIG,
                                                        style={"position": "absolute", "inset": "0"},
                                                    )
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={"flex": "1 1 360px", "minWidth": "320px"},
                                        children=[
                                            html.Div("High-debt risk level", className="graph-title"),
                                            html.Div(
                                                style={
                                                    "width": "min(100%, 460px)",
                                                    "minWidth": "320px",
                                                    "aspectRatio": "16 / 9",
                                                    "position": "relative",
                                                    "margin": "0 auto",
                                                },
                                                children=[
                                                    dcc.Graph(
                                                        id="gauge_debt",
                                                        config=_GRAPH_CONFIG,
                                                        style={"position": "absolute", "inset": "0"},
                                                    )
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="row",
                                children=[
                                    html.Div(
                                        className="card",
                                        style={"flex": "1 1 420px"},
                                        children=[
                                            html.Div("Predicted probabilities", className="graph-title"),
                                            dcc.Graph(
                                                id="prob_bar_chart",
                                                config=_GRAPH_CONFIG,
                                                style={"height": "420px"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="card",
                                        style={"flex": "1 1 420px"},
                                        children=[
                                            html.Div("Risk relative to sample average", className="graph-title"),
                                            dcc.Graph(
                                                id="risk_factor_chart",
                                                config=_GRAPH_CONFIG,
                                                style={"height": "420px"},
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="row",
                                children=[
                                    html.Div(
                                        className="card",
                                        style={"flex": "1 1 420px"},
                                        children=[
                                            html.Div("What-if: borrowing vs high-debt risk", className="graph-title"),
                                            dcc.Graph(
                                                id="what_if_borrowing_debt_chart",
                                                config=_GRAPH_CONFIG,
                                                style={"height": "420px"},
                                            ),
                                        ],
                                    ),
                                ],
                            ),


                            html.Div(
    className="card",
    style={"marginTop": "14px"},
    children=[
        dcc.Markdown(
            textwrap.dedent(
                """
                ### How to interpret this page
                - **Two outcomes:** (1) *Ever defaulted* on a federal student loan, and (2) *Heavy debt burden* defined as **DTI > 0.8** (federal balance divided by estimated annual earnings).
                - **Probabilities are model-based estimates, not guarantees.** They reflect patterns in the HSLS cohort conditional on the selected inputs.
                - **“Risk vs average” is a ratio:** predicted probability divided by the HSLS baseline rate shown at the top of the app. A value of 2.0x means “twice the sample average,” not “twice the dollars.”

                ### What the gauges and percentiles mean
                - **Risk band labels** (Very low, Low, Moderate, High, Very high) are a presentation layer to help interpret a probability. They are not policy thresholds.
                - **Percentile rank** is computed relative to model scores for the HSLS sample. A 90th percentile score means the prediction is higher than 90% of modeled borrowers.

                ### What the “what-if” curve does (and does not do)
                - The borrowing curve isolates the effect of changing **total borrowed** while holding other inputs fixed. It answers: “If everything else stayed the same, how would predicted risk move with borrowing?”
                - This is **not causal**. It does not prove that changing borrowing alone will cause the outcome to change by the same amount.

                ### Important limitations
                - **Default is rare** in HSLS, so even good AUC can still produce many false alarms. Treat default estimates as a coarse risk signal.
                - **Earnings and DTI are constructed** from multiple survey sources and units, so measurement error can affect heavy-debt predictions.
                - HSLS reflects a specific cohort and policy environment; any real deployment would require **recalibration on newer data** and **fairness diagnostics** before use.

                *Proof of concept for an academic project. Do not use for lending, eligibility, or high-stakes decisions.*
                """
            ).strip()
        )
    ],
),
                        ],
                    ),
                ],
            )
        ]
    )

def layout_page_models() -> html.Div:
    children: List = [html.H2("Model comparison and diagnostics")]

    if not model_perf_df.empty:
        percent_cols = [
            c for c in model_perf_df.columns
            if pd.api.types.is_numeric_dtype(model_perf_df[c]) and ("auc" in c.lower() or "ap" in c.lower() or "acc" in c.lower() or "precision" in c.lower() or "recall" in c.lower() or "f1" in c.lower())
        ]

        columns = []
        for c in model_perf_df.columns:
            col = {"name": c, "id": c}
            if c in percent_cols:
                # 0.263 -> 26.3%
                col["type"] = "numeric"
                col["format"] = FormatTemplate.percentage(2)  # 1 decimal place
            columns.append(col)

        children += [
            html.Div(
                className="card",
                children=[
                    html.H3("Cross-validated performance table"),
                    dash_table.DataTable(
                        columns=columns,
                        data=model_perf_df.to_dict("records"),
                        page_size=16,
                        sort_action="native",
                        filter_action="native",
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "fontFamily": "var(--font)",
                            "fontSize": "13px",
                            "padding": "8px",
                            "whiteSpace": "normal",
                            "height": "auto",
                        },
                        style_header={"fontWeight": "700", "backgroundColor": "#f9fafb"},
                    ),
                ],
            )
        ]

        model_col = get_col_case_insensitive(model_perf_df, ["model", "model_name", "name"])
        auc_cols = [c for c in model_perf_df.columns if "auc" in c.lower()]

        if model_col and auc_cols:
            # Melt for bar chart
            plot_df = model_perf_df[[model_col] + auc_cols].copy()
            plot_df = plot_df.melt(id_vars=[model_col], var_name="Metric", value_name="Value")
            fig_auc = px.bar(
                plot_df,
                x=model_col,
                y="Value",
                color="Metric",
                barmode="group",
                title="AUC by model",
                template="jp_clean",
                labels={'Model':''},
            )
            fig_auc.update_layout(height=580, showlegend=False, margin=dict(b=90, r=60),)
            fig_auc.update_xaxes(tickangle=15,)
            children += [
                html.Div(
                    className="card",
                    children=[html.H3("AUC comparison"), dcc.Graph(figure=fig_auc, config=_GRAPH_CONFIG, style={"height": "700px"})],
                )
            ]
    else:
        children += [html.Div(className="card", children=[html.H3("Performance table"), html.Div("No model performance CSV found.")])]

    # Calibration curves
    fig_cal_def_app = build_calibration_fig(cal_default_app_df, "Calibration - default (app features)")
    fig_cal_debt_app = build_calibration_fig(cal_highdebt_app_df, "Calibration - high debt (app features)")
    fig_cal_def_full = build_calibration_fig(cal_default_full_df, "Calibration - default (full features)")
    fig_cal_debt_full = build_calibration_fig(cal_highdebt_full_df, "Calibration - high debt (full features)")

    children += [
        html.Div(
            className="row",
            children=[
                html.Div(className="card", style={"flex": "1 1 520px"}, children=[html.H3("Calibration (app features)"), dcc.Graph(figure=fig_cal_def_app, config=_GRAPH_CONFIG, style={"height": "440px"}), dcc.Graph(figure=fig_cal_debt_app, config=_GRAPH_CONFIG, style={"height": "440px"})]),
                html.Div(className="card", style={"flex": "1 1 520px"}, children=[html.H3("Calibration (full features)"), dcc.Graph(figure=fig_cal_def_full, config=_GRAPH_CONFIG, style={"height": "440px"}), dcc.Graph(figure=fig_cal_debt_full, config=_GRAPH_CONFIG, style={"height": "440px"})]),
            ],
        )
    ]
    SQUARE_MIN_PX = 520   # minimum square size
    SQUARE_MAX_PX = 760   # cap so it doesn't get huge

    def square_graph(fig: go.Figure, min_px: int = SQUARE_MIN_PX, max_px: int = SQUARE_MAX_PX):
        # Ensure the figure isn't pinning its own height
        fig.update_layout(height=None, autosize=True)

        return html.Div(
            style={"display": "flex", "justifyContent": "center", "width": "100%"},
            children=[
                html.Div(
                    style={
                        # responsive width, capped; keep square via aspect-ratio
                        "width": f"min(100%, {max_px}px)",
                        "minWidth": f"{min_px}px",        # guarantees enough room for labels
                        "aspectRatio": "1 / 1",
                        "position": "relative",
                    },
                    children=[
                        dcc.Graph(
                            figure=fig,
                            config=_GRAPH_CONFIG,
                            style={
                                # fill the square wrapper
                                "position": "absolute",
                                "inset": "0",
                            },
                        )
                    ],
                )
            ],
        )

    # ROC / PR
    fig_roc_def = build_roc_fig(roc_default_app_df, roc_default_full_df, "ROC - default")
    fig_roc_debt = build_roc_fig(roc_highdebt_app_df, roc_highdebt_full_df, "ROC - high debt")
    fig_pr_def = build_pr_fig(pr_default_app_df, pr_default_full_df, "Precision-Recall - default")
    fig_pr_debt = build_pr_fig(pr_highdebt_app_df, pr_highdebt_full_df, "Precision-Recall - high debt")

    # Threshold sweeps
    fig_sweep_def = build_sweep_fig(sweep_default_app_df, sweep_default_full_df, "Threshold sweep - default (app vs full)")
    fig_sweep_debt = build_sweep_fig(sweep_highdebt_app_df, sweep_highdebt_full_df, "Threshold sweep - high debt (app vs full)")

    children += [
        html.Div(
            className="card",
            children=[
                html.H3("Discrimination (ROC)"),
                square_graph(fig_roc_def),
                square_graph(fig_roc_debt),
            ],
        ),
        html.Div(
            className="card",
            children=[
                html.H3("Precision-Recall"),
                square_graph(fig_pr_def),
                square_graph(fig_pr_debt),
            ],
        ),
        html.Div(
            className="card",
            children=[
                html.H3("Threshold sweep"),
                square_graph(fig_sweep_def),
                square_graph(fig_sweep_debt),
            ],
        ),
    ]



    children += [
        html.Div(
            className="card",
            children=[
                dcc.Markdown(
                    """
### How to read the diagnostics
- **Calibration:** compares predicted probabilities to observed event rates in bins. A curve near the 45-degree line indicates well-calibrated probabilities.
- **ROC:** shows tradeoffs across all thresholds. It can look strong even when positive events are rare.
- **Precision-Recall:** is often more informative for rare outcomes (like default) because it focuses on performance for the positive class.
- **Threshold sweeps:** show how a metric changes as you move the decision threshold. The best threshold depends on one's objective (minimize false negatives vs false positives).

### App model vs full model
- Charts labeled **“app vs full”** compare an 8-feature “student-facing” specification to a richer specification used for benchmarking.
- The app model is designed for **interpretability and usability**, so it may trade off predictive performance.

### Cautions for interpretation
- A single metric does not define usefulness. For default, low prevalence means **precision will typically be low** at conventional thresholds.
- Use these plots to justify why the app should display **probabilities and relative risk** rather than hard classifications.

*Diagnostics summarize out-of-sample cross-validation results from the HSLS modeling sample. They do not guarantee performance for other cohorts or future years.*
""".strip()
                )
            ],
        )
    ]

    return html.Div(children)

def layout_page_pca() -> html.Div:
    return html.Div(
        [
            html.H2("Borrower profiles in a lower-dimensional space (PCA)"),
            html.Div(
                className="card",
                children=[
                    html.H3("PCA embeddings"),
                    html.Div(
                        className="row",
                        children=[
                            html.Div(
                                style={"flex": "1 1 250px", "minWidth": "240px"},
                                children=[
                                    html.Div("Color by", className="control-label"),
                                    dcc.Dropdown(
                                        id="pca-color",
                                        options=[
                                            {"label": "Cluster", "value": "cluster"},
                                            {"label": "Default", "value": TARGET_DEFAULT},
                                            {"label": "High debt burden", "value": TARGET_DEBT},
                                        ],
                                        value="cluster",
                                        clearable=False,
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Graph(id="fig-pca", config=_GRAPH_CONFIG, style={"height": "580px"}),
                    dcc.Graph(id="fig-pca-var", config=_GRAPH_CONFIG, style={"height": "440px"}),
                    html.H3("Cluster summary"),
                    dash_table.DataTable(
                        id="tbl-cluster",
                        columns=[{"name": c, "id": c} for c in (cluster_summary_df.columns.tolist() if not cluster_summary_df.empty else [])],
                        data=(cluster_summary_df.to_dict("records") if not cluster_summary_df.empty else []),
                        page_size=12,
                        sort_action="native",
                        filter_action="native",
                        style_table={"overflowX": "auto"},
                        style_cell={"fontFamily": "var(--font)", "fontSize": "12px", "padding": "8px", "whiteSpace": "normal", "height": "auto"},
                        style_header={"fontWeight": "700", "backgroundColor": "#f9fafb"},
                    ),
                ],
            ),
            html.Div(
                className="card",
                children=[
                    dcc.Markdown(
                        """
### What this page shows
- **PCA embeds borrower profiles** into two dimensions (PC1 and PC2) to visualize broad similarity patterns in the underlying feature space.
- Clusters (if selected) are **unsupervised groupings**. They are descriptive summaries, not “types of people” with fixed meaning.

### How to interpret structure
- If points separate by outcome color, it suggests the features contain signal related to that outcome.
- If points overlap heavily, it suggests the outcome is not easily separable in this representation (or the separation is non-linear and not captured by PCA).

### Important caveats
- PCA axes are linear combinations of features. **PC1 and PC2 are not single interpretable variables.**
- Cluster labels (1, 2, 3, …) are arbitrary and depend on preprocessing choices.
- This visualization is **not causal** and should not be used to infer individual-level mechanisms.

*Use PCA as a qualitative tool to sanity-check the feature space and communicate heterogeneity in the sample.*
""".strip()
                    )
                ],
            ),
        ]
    )

def layout_page_exploratory() -> html.Div:
    children: List = [html.H2("Exploring default and high debt burden in the HSLS sample")]

    if df_joint.empty:
        return html.Div(children + [html.Div(className="card", children=["No joint-sample rows available (df_both is empty)."])])

    contingency = pd.crosstab(df_joint["default_status"], df_joint["debt_status"])
    contingency_props = contingency / max(contingency.to_numpy().sum(), 1)

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=contingency_props.values,
            x=contingency_props.columns,
            y=contingency_props.index,
            coloraxis="coloraxis",
            hovertemplate="Default: %{y}<br>Debt: %{x}<br>Proportion: %{z:.2%}<extra></extra>",
        ),
    )
    fig_heat.update_layout(
        title="Joint distribution of default and heavy debt burden (proportions)",
        xaxis_title="",
        yaxis_title="",
        coloraxis=dict(colorscale="Blues"),
        template="jp_clean",
        height=460,
    )
    fig_heat.update_traces(
        text=contingency_props.values,
        texttemplate="%{text:.2%}",
        hovertemplate=None,
    )

    children += [html.Div(className="card", children=[html.H3("Joint distribution"), dcc.Graph(figure=fig_heat, config=_GRAPH_CONFIG, style={"height": "480px"})])]


    if "X1FAMINCOME" in df_joint.columns:
        # Plot using the plot column
        fig_fam = px.histogram(
            df_joint,
            x="X1FAMINCOME_plot",
            color="default_debt_group",
            barmode="group",
            #histnorm="percent",
            category_orders={"X1FAMINCOME_plot": income_short_order},
            labels={"X1FAMINCOME_plot": short_label("X1FAMINCOME"), "default_debt_group": "Group"},
            title="Baseline family income distribution by default/debt group",
            template="jp_clean",
        )

        # Make Plotly respect order even if anything gets cast back to object
        fig_fam.update_xaxes(categoryorder="array", categoryarray=income_short_order)
        ticktext_no_prefix = [
            s.replace("Family income ", "", 1).strip() if isinstance(s, str) else s
            for s in income_short_order
            ]
        fig_fam.update_xaxes(
            tickmode="array",
            tickvals=income_short_order,
            ticktext=ticktext_no_prefix,
            tickangle=30,
            automargin=True,
            title_standoff=3
        )

        fig_fam.update_layout(height=520, margin=dict(b=120))

        children += [
            html.Div(
                className="card",
                children=[
                    html.H3("Income by group"),
                    dcc.Graph(figure=fig_fam, config=_GRAPH_CONFIG, style={"height": "700px"}),
                ],
            )
        ]


    if "annual_earnings" in df_joint.columns:
        fig_earn = px.box(
            df_joint,
            x="default_debt_group",
            y="annual_earnings",
            labels={"default_debt_group": "Group", "annual_earnings": short_label("annual_earnings")},
            title="Annual earnings by default/debt group",
            template="jp_clean",
        )
        fig_earn.update_layout(height=520)
        children += [html.Div(className="card", children=[html.H3("Earnings by group"), dcc.Graph(figure=fig_earn, config=_GRAPH_CONFIG, style={"height": "540px"})])]

    if "institution_type" in df_joint.columns:
        inst_order = sorted(df_joint["institution_type"].dropna().unique().tolist())
        fig_inst = px.histogram(
            df_joint,
            x="institution_type",
            color="default_debt_group",
            barmode="group",
            #histnorm="percent",
            category_orders={"institution_type": inst_order},
            labels={"institution_type": short_label("institution_type"), "default_debt_group": "Group"},
            title="Institution type by default/debt group",
            template="jp_clean",
        )
        fig_inst.update_layout(height=600)
        fig_inst.update_xaxes(tickangle=0, tickfont_size=10)
        children += [html.Div(className="card", children=[html.H3("Institution type by group"), dcc.Graph(figure=fig_inst, config=_GRAPH_CONFIG, style={"height": "700px"})])]

    children += [
        html.Div(
            className="card",
            children=[
                dcc.Markdown(
                    """
#### What this page is (and is not)
- These plots are **descriptive** comparisons across groups in the HSLS sample, such as default vs high-debt combinations.
- They help answer: “Which observable characteristics differ across outcome groups?” They do **not** answer: “What causes default or high debt?”

### Reading the joint distribution
- A large “High debt burden / No default” group implies that heavy debt and default are related but distinct. Many borrowers experience high DTI without entering default.

### Reading grouped distributions
- Differences by income, institution type, or earnings can reflect many factors (selection into colleges, labor market variation, unobserved support systems).

*Group-level correlations should not be used as individual-level rules. For higher-stakes use, add subgroup calibration and fairness diagnostics.*
""".strip()
                )
            ],
        )
    ]
    return html.Div(children)

def layout_page_features() -> html.Div:
    children: List = [html.H2("Which features are most associated with risk? (logistic app models)")]

    # Default coefficients
    if not coef_default.empty:
        df_cd = coef_default.copy()
        df_cd["direction"] = np.where(df_cd["Coefficient"] >= 0, "Increases default risk", "Decreases default risk")
        df_cd["Feature_readable"] = df_cd["Feature"].apply(make_feature_readable)
        df_cd = df_cd.sort_values("Coefficient", ascending=False)

        fig_coef_def = px.bar(
            df_cd.head(10).sort_values("Coefficient"),
            x="Coefficient",
            y="Feature_readable",
            color="direction",
            orientation="h",
            labels={"Coefficient": "Log-odds coefficient", "Feature_readable": ""},
            title="Top features associated with higher probability of ever defaulting",
            template="jp_clean",
        )
        fig_coef_def.update_layout(yaxis=dict(automargin=True), height=560, showlegend=False)
        children.append(html.Div(className="card", children=[html.H3("Default (top coefficients)"), dcc.Graph(figure=fig_coef_def, config=_GRAPH_CONFIG, style={"height": "580px"})]))
    else:
        children.append(html.Div(className="card", children=[html.H3("Default coefficients"), html.Div("Coefficients CSV not available.")]))

    # High-debt coefficients
    if not coef_debt.empty:
        df_ch = coef_debt.copy()
        df_ch["direction"] = np.where(df_ch["Coefficient"] >= 0, "Increases heavy-debt risk", "Decreases heavy-debt risk")
        df_ch["Feature_readable"] = df_ch["Feature"].apply(make_feature_readable)
        df_ch = df_ch.sort_values("Coefficient", ascending=False)

        fig_coef_debt = px.bar(
            df_ch.head(10).sort_values("Coefficient"),
            x="Coefficient",
            y="Feature_readable",
            color="direction",
            orientation="h",
            labels={"Coefficient": "Log-odds coefficient", "Feature_readable": ""},
            title="Top features associated with higher probability of heavy debt burden",
            template="jp_clean",
        )
        fig_coef_debt.update_layout(yaxis=dict(automargin=True), height=560, showlegend=False)
        children.append(html.Div(className="card", children=[html.H3("High debt burden (top coefficients)"), dcc.Graph(figure=fig_coef_debt, config=_GRAPH_CONFIG, style={"height": "580px"})]))
    else:
        children.append(html.Div(className="card", children=[html.H3("High-debt coefficients"), html.Div("Coefficients CSV not available.")]))

    # Scatter: default vs debt coefficients (merge by Feature)
    if (not coef_default.empty) and (not coef_debt.empty):
        df_cd_small = coef_default.rename(columns={"Feature": "Feature", "Coefficient": "coef_default"})
        df_ch_small = coef_debt.rename(columns={"Feature": "Feature", "Coefficient": "coef_debt"})
        merged = df_cd_small.merge(df_ch_small, on="Feature", how="inner")
        if not merged.empty:
            merged["Feature_readable"] = merged["Feature"].apply(make_feature_readable)
            merged["importance_sum"] = merged["coef_default"].abs() + merged["coef_debt"].abs()
            merged_top = merged.sort_values("importance_sum", ascending=False).head(40)

            fig_scatter = px.scatter(
                merged_top,
                x="coef_default",
                y="coef_debt",
                hover_name="Feature_readable",
                labels={"coef_default": "Coefficient (default)", "coef_debt": "Coefficient (high debt)"},
                title="Feature effects: default vs heavy debt (app logistic models)",
                template="jp_clean",
            )
            axis_min = float(min(merged_top["coef_default"].min(), merged_top["coef_debt"].min()))
            axis_max = float(max(merged_top["coef_default"].max(), merged_top["coef_debt"].max()))
            fig_scatter.add_shape(type="line", x0=axis_min, y0=axis_min, x1=axis_max, y1=axis_max, line=dict(dash="dash"))
            fig_scatter.update_layout(height=560)
            children.append(html.Div(className="card", children=[html.H3("Default vs high-debt coefficients"), dcc.Graph(figure=fig_scatter, config=_GRAPH_CONFIG, style={"height": "700px"})]))

    # Permutation importance
    if not perm_import_default.empty:
        dfp = perm_import_default.sort_values("importance", ascending=False).head(20)
        fig_perm = px.bar(
            dfp.sort_values("importance"),
            x="importance",
            y="feature_readable",
            orientation="h",
            labels={"importance": "Permutation importance", "feature_readable": "Feature"},
            title="Permutation importance (default app model)",
            template="jp_clean",
        )
        fig_perm.update_layout(yaxis=dict(automargin=True), height=560)
        children.append(html.Div(className="card", children=[html.H3("Permutation importance"), dcc.Graph(figure=fig_perm, config=_GRAPH_CONFIG, style={"height": "580px"})]))

    children.append(
        html.Div(
            className="card",
            children=[
                dcc.Markdown(
                    """
### Interpreting coefficients and feature importance
- Logistic coefficients are **log-odds effects** conditional on the other model inputs. Signs indicate direction; magnitudes indicate strength on the log-odds scale.
- One-hot encoded categories are interpreted **relative to the omitted reference category**. A positive coefficient means higher predicted risk compared to that baseline.

### Practical guidance
- Use this page to explain the model in plain language: which inputs matter most, and whether they increase or decrease predicted risk.
- Avoid over-interpreting small differences. Coefficients can shift with regularization, correlated predictors, and missing-data patterns.

*These feature effects are intended for interpretability in a student-facing tool, not for policy or enforcement decisions.*
""".strip()
                )
            ],
        )
    )
    return html.Div(children)

def layout_page_data() -> html.Div:
    meta_rows = [
        {"Item": "Rows (full df)", "Value": f"{len(df):,}"},
        {"Item": "Rows (default-model eligible)", "Value": f"{len(df_t1):,}"},
        {"Item": "Rows (high-debt-model eligible)", "Value": f"{len(df_t2):,}"},
        {"Item": "Baseline default rate", "Value": _fmt_pct(baseline_default_rate)},
        {"Item": "Baseline high-debt rate", "Value": _fmt_pct(baseline_high_debt_rate)},
    ]
    meta_df = pd.DataFrame(meta_rows)

    sample_cols = ["STU_ID"] + [c for c in APP_FEATURES if c in df.columns] + [TARGET_DEFAULT, TARGET_DEBT]
    sample_cols = [c for c in sample_cols if c in df.columns]
    sample_df = df[sample_cols].head(200).copy()

    return html.Div(
        [
            html.H2("Data overview"),
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="card",
                        style={"flex": "1 1 420px"},
                        children=[
                            html.H3("Dataset summary"),
                            dash_table.DataTable(
                                columns=[{"name": c, "id": c} for c in meta_df.columns],
                                data=meta_df.to_dict("records"),
                                style_cell={"fontFamily": "var(--font)", "fontSize": "12px", "padding": "8px"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f9fafb"},
                            ),
                        ],
                    ),
                    html.Div(
                        className="card",
                        style={"flex": "2 1 640px"},
                        children=[
                            html.H3("Sample rows"),
                            dash_table.DataTable(
                                columns=[{"name": c, "id": c} for c in sample_df.columns],
                                data=sample_df.to_dict("records"),
                                page_size=12,
                                sort_action="native",
                                filter_action="native",
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "fontFamily": "var(--font)",
                                    "fontSize": "12px",
                                    "padding": "8px",
                                    "whiteSpace": "normal",
                                    "height": "auto",
                                },
                                style_header={"fontWeight": "700", "backgroundColor": "#f9fafb"},
                            ),
                        ],
                    ),
                ],
            ),

            html.Div(
                className="card",
                children=[
                    dcc.Markdown(
                        """
### Data notes
- Source: **HSLS 2009 Postsecondary Education Transcript Study (PETS)** public-use borrower sample.
- Borrowers are defined using positive cumulative federal borrowing and/or positive outstanding federal balance.
- The heavy-debt outcome uses a constructed DTI measure: **DTI = federal balance / estimated annual earnings**, with heavy burden defined as **DTI > 0.8**.

### Cleaning and missingness
- HSLS missing codes are mapped to null values. For modeling, numeric fields use median imputation; categorical fields use most-frequent imputation with one-hot encoding.
- Earnings are assembled from multiple survey items and pay units, using a documented fallback hierarchy. This introduces measurement error that can affect DTI.

### Responsible use
- This dataset reflects a particular cohort and economic context. Predictions may not transport to newer cohorts without re-estimation or recalibration.
- Before any real-world deployment, add privacy review, subgroup diagnostics, and fairness evaluation.

*This page provides transparency about what is in the dataset and what transformations were required to make the modeling sample.*
                        """.strip()
                    )
                ],
            ),
        ]
    )


# ------------------------------------------------------------
# Dash app + routing
# ------------------------------------------------------------
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(
    className="container",
    children=[
        dcc.Location(id="url", refresh=False),
        html.Div(id="top-header"),
        html.Div(id="page-content"),
        html.Div("INDENG 242A Project. UC Berkeley. Proof of Concept only. Do not rely on it for any decision making.", className="footer-note"),
    ],
)

app.validation_layout = html.Div(
    className="container",
    children=[
        dcc.Location(id="url", refresh=False),
        layout_header("/"),
        layout_page_risk_explorer(),
        layout_page_models(),
        layout_page_pca(),
        layout_page_exploratory(),
        layout_page_features(),
        layout_page_data(),
        dcc.Graph(id="prob_bar_chart"),
        dcc.Graph(id="risk_factor_chart"),
        dcc.Graph(id="what_if_borrowing_debt_chart"),
        dcc.Graph(id="what_if_borrowing_default_chart"),
        dcc.Graph(id="log_odds_borrowing_chart"),
        dcc.Graph(id="gauge_default"),
        dcc.Graph(id="gauge_debt"),
        html.Div(id="prob_summary"),
        html.Div(id="kpi_row"),
        dcc.Dropdown(id="pca-color"),
        dcc.Graph(id="fig-pca"),
        dcc.Graph(id="fig-pca-var"),
        dash_table.DataTable(id="tbl-cluster"),
    ],
)

@app.callback(Output("top-header", "children"), Input("url", "pathname"))
def _render_header(pathname: str):
    return layout_header(pathname or "/")

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname: str):
    pathname = pathname or "/"
    if pathname == "/models":
        return layout_page_models()
    if pathname == "/pca":
        return layout_page_pca()
    if pathname == "/exploratory":
        return layout_page_exploratory()
    if pathname == "/features":
        return layout_page_features()
    if pathname == "/data":
        return layout_page_data()
    return layout_page_risk_explorer()

# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------
@app.callback(Output("input_S3CLGCOST", "marks"), Input("input_S3CLGCOST", "value"))
def update_cost_marks(current_value):
    marks = {float(cost_min): money_label(cost_min), float(cost_max): money_label(cost_max)}
    if current_value is not None and cost_min < float(current_value) < cost_max:
        marks[float(current_value)] = money_label(float(current_value))
    items = sorted(marks.items(), key=lambda kv: kv[0])
    return {k: v for k, v in items}

@app.callback(
        [
        Output("kpi_row", "children"),
        Output("gauge_default", "figure"),
        Output("gauge_debt", "figure"),
        Output("prob_bar_chart", "figure"),
        Output("risk_factor_chart", "figure"),
        Output("what_if_borrowing_debt_chart", "figure"),
    ],

    [
        Input("input_S3CLGBORROW", "value"),
        Input("input_S3CLGCOST", "value"),
        Input("input_intended_field_group", "value"),
        Input("input_institution_type", "value"),
        Input("input_X1FAMINCOME", "value"),
        Input("input_X1PAREDU", "value"),
        Input("input_X1SEX", "value"),
        Input("input_S3CLGSEL", "value"),
    ],
)
def update_predictions(borrow, cost, intended_field_group, institution_type, fam_income, parent_edu, sex, selectivity):
    input_dict = {
        "S3CLGBORROW": float(_safe_float(borrow, 0.0)),
        "S3CLGCOST": float(_safe_float(cost, 0.0)),
        "intended_field_group": intended_field_group,
        "institution_type": institution_type,
        "X1FAMINCOME": fam_income,
        "X1PAREDU": parent_edu,
        "X1SEX": sex,
        "S3CLGSEL": selectivity,
    }
    input_df = pd.DataFrame([input_dict])

    preds = predict_student_risks_app(input_df)
    row = preds.iloc[0]

    if np.isnan(row.get("prob_default", np.nan)) or np.isnan(row.get("prob_high_debt", np.nan)):
        kpis = [
            _kpi("Default probability", "—"),
            _kpi("Default vs average", "—"),
            _kpi("High-debt probability", "—"),
            _kpi("High-debt vs average", "—"),
        ]
        summary = html.Div(
            [
                html.Div("Predictions unavailable because models did not load.", style={"fontWeight": "700"}),
                html.Div("Check that .pkl model files exist in /data and are compatible with this feature schema.", className="small-note"),
            ]
        )
        blank = _warn_fig("Predictions unavailable (model not loaded).", "Not available")
        blank_gauge = _warn_fig("Predictions unavailable (model not loaded).", "Not available")
        blank = _warn_fig("Predictions unavailable (model not loaded).", "Not available")

        return kpis, blank_gauge, blank_gauge, blank, blank, blank


    # Percentiles
    pct_def = _percentile(oof_default_scores, float(row["prob_default"]))
    pct_debt = _percentile(oof_highdebt_scores, float(row["prob_high_debt"]))

    kpis = [
        _kpi("Default probability", f"{row['prob_default']:.1%}"),
        _kpi("Default vs average", f"{row['default_risk_vs_average']:.1f}x"),
        _kpi("High-debt probability", f"{row['prob_high_debt']:.1%}"),
        _kpi("High-debt vs average", f"{row['high_debt_risk_vs_average']:.1f}x"),
    ]
    if pct_def is not None:
        kpis[0] = _kpi("Default probability", f"{row['prob_default']:.1%} (pctl {pct_def:.0%})")
    if pct_debt is not None:
        kpis[2] = _kpi("High-debt probability", f"{row['prob_high_debt']:.1%} (pctl {pct_debt:.0%})")
    # -------------------------
    # Gauges (equal-size bands)
    # -------------------------
    p_def = float(row["prob_default"])
    p_debt = float(row["prob_high_debt"])

    axis_def = _prob_to_equal_band_axis(p_def, DEFAULT_RISK_BANDS)
    axis_debt = _prob_to_equal_band_axis(p_debt, DEBT_RISK_BANDS)

    tickvals_def = [i + 0.5 for i in range(len(DEFAULT_RISK_BANDS))]
    ticktext_def = [b[0] for b in DEFAULT_RISK_BANDS]

    tickvals_debt = [i + 0.5 for i in range(len(DEBT_RISK_BANDS))]
    ticktext_debt = [b[0] for b in DEBT_RISK_BANDS]

    fig_gauge_def = go.Figure(
        go.Indicator(
            mode="gauge",
            value=axis_def,
            gauge={
                "axis": {
                    "range": [0, len(DEFAULT_RISK_BANDS)],
                    "tickmode": "array",
                    "tickvals": tickvals_def,
                    "ticktext": ticktext_def,
                    "tickfont": {"size": 12},
                },
                "bar": {"color": "rgba(0,0,0,0)"},  # hide fill bar; we use threshold as the pointer
                "steps": [
                    {"range": [i, i + 1], "color": DEFAULT_RISK_COLORS[i]}
                    for i in range(len(DEFAULT_RISK_BANDS))
                ],
                # This line is the "needle/pointer"
                "threshold": {
                    "line": {"color": "#111827", "width": 7},
                    "thickness": 0.85,
                    "value": axis_def,
                },
            },
        )
    )
    fig_gauge_def.update_layout(
        template="jp_clean",
        margin=dict(l=60, r=60, t=40, b=20),
    )
    fig_gauge_def.add_annotation(
        x=0.5, y=0.17, xref="paper", yref="paper", showarrow=False, align="center",
        text=(
            f"<b>{p_def:.1%}</b><br>"
            f"<span style='color:#6b7280'>{risk_band_default(p_def)}</span><br>"
            f"<span style='color:#6b7280;font-size:0.9em'>"
            f"{row['default_risk_vs_average']:.1f}x baseline"
            f"</span>"
        ),
        font=dict(size=16),
    )

    fig_gauge_debt = go.Figure(
        go.Indicator(
            mode="gauge",
            value=axis_debt,
            gauge={
                "axis": {
                    "range": [0, len(DEBT_RISK_BANDS)],
                    "tickmode": "array",
                    "tickvals": tickvals_debt,
                    "ticktext": ticktext_debt,
                    "tickfont": {"size": 12},
                },
                "bar": {"color": "rgba(0,0,0,0)"},
                "steps": [
                    {"range": [i, i + 1], "color": DEBT_RISK_COLORS[i]}
                    for i in range(len(DEBT_RISK_BANDS))
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 7},
                    "thickness": 0.85,
                    "value": axis_debt,
                },
            },
        )
    )
    fig_gauge_debt.update_layout(
        template="jp_clean",
        margin=dict(l=30, r=60, t=40, b=20),
    )
    fig_gauge_debt.add_annotation(
        x=0.5, y=0.17, xref="paper", yref="paper", showarrow=False, align="center",
        text=(
            f"<b>{p_debt:.1%}</b><br>"
            f"<span style='color:#6b7280'>{risk_band_debt(p_debt)}</span><br>"
            f"<span style='color:#6b7280;font-size:0.9em'>"
            f"{row['high_debt_risk_vs_average']:.1f}x baseline"
            f"</span>"
        ),
        font=dict(size=16),
    )


    # Bar: predicted probabilities
    fig_probs = px.bar(
        x=["Ever defaulted", "Heavy debt burden"],
        y=[row["prob_default"], row["prob_high_debt"]],
        labels={"x": "", "y": "Predicted probability"},
        range_y=[0, 1],
        template="jp_clean",
        title="Predicted probabilities for this student",
    )
    fig_probs.update_traces(text=[f"{row['prob_default']:.1%}", f"{row['prob_high_debt']:.1%}"], textposition="outside")
    fig_probs.update_layout(height=420)

    # Bar: risk factors vs average
    fig_factor = px.bar(
        x=["Ever defaulted", "Heavy debt burden"],
        y=[row["default_risk_vs_average"], row["high_debt_risk_vs_average"]],
        labels={"x": "", "y": "Risk relative to sample average (1 = average)"},
        template="jp_clean",
        title="Risk relative to average student in the HSLS sample",
    )
    fig_factor.add_hline(y=1.0, line_dash="dash")
    fig_factor.update_traces(
        text=[f"{row['default_risk_vs_average']:.1f}x", f"{row['high_debt_risk_vs_average']:.1f}x"],
        textposition="outside",
    )
    fig_factor.update_layout(height=420)

    # What-if curves over borrowing
    borrow_grid = np.linspace(max(0, borrow_min), borrow_max, 40)
    scenario_df = pd.DataFrame([{**input_dict, "S3CLGBORROW": float(b)} for b in borrow_grid])
    scenario_preds = predict_student_risks_app(scenario_df)

    fig_what_if_debt = px.line(
        x=borrow_grid,
        y=scenario_preds["prob_high_debt"],
        labels={"x": short_label("S3CLGBORROW"), "y": "Predicted probability"},
        template="jp_clean",
        title="Borrowing vs high-debt risk (holding other inputs fixed)",
    )
    fig_what_if_debt.add_vline(x=float(input_dict["S3CLGBORROW"]), line_dash="dash", annotation_text="Current choice")
    fig_what_if_debt.update_yaxes(range=[0, 1])
    fig_what_if_debt.update_layout(height=420)

    fig_what_if_default = px.line(
        x=borrow_grid,
        y=scenario_preds["prob_default"],
        labels={"x": short_label("S3CLGBORROW"), "y": "Predicted probability"},
        template="jp_clean",
        title="Borrowing vs default risk (holding other inputs fixed)",
    )
    fig_what_if_default.add_vline(x=float(input_dict["S3CLGBORROW"]), line_dash="dash", annotation_text="Current choice")
    fig_what_if_default.update_yaxes(range=[0, 1])
    fig_what_if_default.update_layout(height=420)

    

    return kpis, fig_gauge_def, fig_gauge_debt, fig_probs, fig_factor, fig_what_if_debt

@app.callback(
    [Output("fig-pca", "figure"), Output("fig-pca-var", "figure")],
    [Input("pca-color", "value")],
)
def update_pca(color_by: str):
    fig1 = fig_pca_scatter(color_by or "cluster")
    fig2 = fig_pca_variance()
    return fig1, fig2

# ------------------------------------------------------------
# 17) Local run
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)