import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from engine import project_cashflows
import plotly.graph_objects as go

st.set_page_config(page_title="Loan Portfolio Cashflow Forecaster", layout="wide")

st.title("📊 Loan Portfolio Cashflow Forecaster")

# --- Helpers ---
def _npv(cashflows, r_m):
    return sum(cf / ((1 + r_m) ** t) for t, cf in enumerate(cashflows, start=1))
def _wal(months_series, principal_series):
    total_prin = principal_series.sum()
    if total_prin <= 0: return 0.0
    return float((months_series * principal_series).sum() / total_prin)
def _wal_with_recov(months_series, principal_series, recoveries_series):
    total = (principal_series + recoveries_series).sum()
    if total <= 0: return 0.0
    return float((months_series * (principal_series + recoveries_series)).sum() / total)

# --- Sidebar inputs ---
with st.sidebar:
    st.header("Inputs")
    file = st.file_uploader("Upload loan tape (.xlsx)", type=["xlsx"])
    fname = getattr(file, "name", None)
    if fname and st.session_state.get("last_file_name") not in (None, fname):
        st.session_state.pop("monthly_df", None); st.session_state.pop("by_loan", None)
    if fname: st.session_state["last_file_name"] = fname

    st.divider()
    st.subheader("Assumptions")
    # --- Scenario presets (Germany) ---
    st.markdown("**Scenario presets (Germany)**")
    scenarios = {
        "Custom (no preset)": None,
        "Germany – Baseline": {
            "WA_PD_annual_ref": 0.0250,
            "WA_LGD_ref": 0.725,
            "CPR_annual_ref": 0.22,
            "Servicing_bps_ref": 100.0,
            "Recovery_lag_m_ref": 12
        },
        "Germany – Adverse": {
            # PD +40%, LGD +25%, CPR ↓ to 12%, Recovery lag ↑ to 18m
            "WA_PD_annual_ref": 0.0250 * 1.40,
            "WA_LGD_ref": 0.725 * 1.25,
            "CPR_annual_ref": 0.12,
            "Servicing_bps_ref": 100.0,
            "Recovery_lag_m_ref": 18
        },
        "Germany – Severe": {
            # PD +100%, LGD +50%, CPR ↓ to 6%, Recovery lag ↑ to 24m
            "WA_PD_annual_ref": 0.0250 * 2.00,
            "WA_LGD_ref": 0.725 * 1.50,
            "CPR_annual_ref": 0.06,
            "Servicing_bps_ref": 100.0,
            "Recovery_lag_m_ref": 24
        }
    }
    sel_scn = st.selectbox("Scenario preset (applies defaults below)", list(scenarios.keys()))
    if sel_scn != "Custom (no preset)":
        scn_vals = scenarios[sel_scn]
        # Apply scenario defaults into session (without forcing over user edits)
        for k, v in scn_vals.items():
            st.session_state[k] = float(v) if isinstance(v, (int, float)) else v
        # Ensure PD/LGD come from scenario, not tape, unless user chooses otherwise
        st.session_state["use_wa_from_tape"] = False
        st.info(f"Applied **{sel_scn}** defaults. PD/LGD are set from scenario (uncheck to override).")
    st.caption("Tip: Scenario presets set *default* values for PD/LGD/CPR/fees/recovery lag. You can still edit them below.")

    months = st.slider("Projection horizon (months)", 12, 120, 60, step=6)
    disc_rate_annual = st.number_input("Discount rate (annual, %)", 0.0, 100.0, 8.0, 0.25, format="%.2f")/100.0
    
    # Use ref values as defaults if present (without overriding tape auto-setting)
    def _get_ref(name, fallback):
        return float(st.session_state.get(name, fallback))
    
    use_wa_from_tape = st.checkbox("Auto-set WA PD/LGD from tape (OB-weighted)", value=True, key="use_wa_from_tape")
    WA_PD_annual = st.number_input("WA PD (annual, %)", 0.0, 100.0, _get_ref("WA_PD_annual_ref", 2.5), 0.05, format="%.3f")/100.0
    WA_LGD = st.number_input("WA LGD (%)", 0.0, 100.0, _get_ref("WA_LGD_ref", 72.5), 0.5, format="%.3f")/100.0
    CPR_annual = st.number_input("CPR (annual, %)", 0.0, 100.0, _get_ref("CPR_annual_ref", 22.0), 0.5, format="%.3f")/100.0
    Servicing_bps = st.number_input("Servicing fee (bps p.a., on opening balance)", 0.0, 500.0, _get_ref("Servicing_bps_ref", 100.0), 5.0, format="%.1f")
    Recovery_lag_m = int(st.number_input("Recovery lag (months)", 0, 60, int(_get_ref("Recovery_lag_m_ref", 12)), 1))

    # 🔎 One-line mechanics explainer (how the engine applies assumptions each month)
    st.caption("Mechanics: each month we apply scheduled amortisation → expected defaults → expected prepayments; recoveries arrive after the lag; servicing is charged on opening balance; net cash = interest + principal + prepay + recoveries − servicing.")

    run_clicked = st.button("⚙️ Run / Re-run projection")
    clear_clicked = st.button("🗑️ Clear current projection")
    if clear_clicked:
        st.session_state.pop("monthly_df", None); st.session_state.pop("by_loan", None)
    
    st.divider()

    # 📚 Research on Germany (moved into an expander)
    with st.expander("📚 Research on Germany", expanded=False):
        # --- Baseline references ---
        st.markdown("**Baseline references**")
        st.markdown("- DBRS Morningstar (2024): Lifetime default **11.9%**; expected recovery **27.5%** → LGD **72.5%**. [take me there](https://dbrs.morningstar.com/research/427856/morningstar-dbrs-finalises-provisional-credit-ratings-on-fortuna-consumer-loan-abs-2024-1-designated-activity-company)")
        st.markdown("- Fitch (2020): Example CPR for German consumer ABS ~**22%**. [take me there](https://www.fitchratings.com/research/structured-finance/fitch-assigns-sc-germany-sa-compartment-consumer-2020-1-expected-ratings-28-09-2020)")
        st.markdown("- EBA Loan Enforcement Benchmarks (2020): Germany recoveries typically **0.6–1.3 years** → **12 months** used. [take me there](https://www.eba.europa.eu/sites/default/files/document_library/About%20Us/Missions%20and%20tasks/Call%20for%20Advice/2020/Report%20on%20the%20benchmarking%20of%20national%20loan%20enforcement%20frameworks/962022/Report%20on%20the%20benchmarking%20of%20national%20loan%20enforcement%20frameworks.pdf)")
        st.markdown("- ECB supervisory data (2024): Household loan NPL ratio in Germany around **2.2%**, anchoring baseline PD. [take me there](https://www.banque-france.fr/system/files/2025-03/ssm.pr250320.pdf)")

        # --- Adverse references ---
        st.markdown("**Adverse references**")
        st.markdown("- Bundesbank Bank Lending Survey (Jan 2025): Elevated NPLs tightening credit standards for households and consumer credit. [take me there](https://www.bundesbank.de/en/press/press-releases/january-results-of-the-bank-lending-survey-bls-in-germany-950082)")
        st.markdown("- ECB/EBA stress testing practice: Adverse scenarios often apply **PD uplifts** with moderate LGD add-ons. [take me there](https://www.eba.europa.eu/risk-analysis-and-data/eu-wide-stress-testing)")
        st.markdown("- EBA Guidelines (2017/2019): Require **downturn LGD adjustments** above long-run averages. [take me there](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/model-validation/guidelines-pd-estimation-lgd)")

        # --- Severe references ---
        st.markdown("**Severe references**")
        st.markdown("- ECB (June 2025): Supervisors warned banks to watch for a **rise in bad loans**, particularly in consumer credit. [take me there](https://www.reuters.com/business/finance/ecb-tells-banks-again-watch-out-rise-bad-loans-2025-06-18/)")
        st.markdown("- Bundesbank (Nov 2024): Warned of **elevated corporate and consumer default risk** under stress conditions. [take me there](https://www.reuters.com/markets/europe/germany-faces-high-corporate-default-risk-2025-bundesbank-says-2024-11-21/)")
        st.markdown("- Basel III / EBA downturn LGD: Severe stress often calibrated with **large PD uplifts** and **downturn LGD add-ons**. [take me there](https://www.bis.org/publ/bcbs115.htm)")

    st.divider()
    st.subheader("Regulatory checks (upload file to run)")

# --- Load data ---
if file is None:
    st.info("Upload your loan tape (.xlsx) to begin."); st.stop()

try:
    loans = pd.read_excel(file, sheet_name="loan_tape")
except Exception as e:
    st.error(f"Could not read 'loan_tape' sheet: {e}"); st.stop()

# Validate strict schema
expected_cols = ["loan_id","opening_balance","interest_rate (annual)","maturity_months","months_elapsed","monthly_rate","remaining_term","monthly_payment","sched_principal_m1","pd","lgd"]
missing = [c for c in expected_cols if c not in loans.columns]
if missing:
    st.error(f"The tape is missing required columns: {missing}")
    st.stop()

st.subheader("📄 Loan tape preview")
st.dataframe(loans.head(10).set_index("loan_id"), use_container_width=True)

# --- Data validation & exclusion rules ---
st.subheader("🧹 Data validation")
min_ob = st.number_input("Minimum opening balance to include (€)", min_value=0.0, value=0.0, step=1.0, help="Rows with opening_balance ≤ this value will be excluded from all calculations.")
exclude_zero = st.checkbox("Exclude rows with opening_balance ≤ minimum", value=True)

# Coerce OB to numeric for validation
ob_num_all = pd.to_numeric(loans["opening_balance"], errors="coerce").fillna(0.0)
ex_mask = (ob_num_all <= min_ob) if exclude_zero else (ob_num_all < 0)  # if not excluding, only drop negatives
excluded_df = loans.loc[ex_mask].copy()
loans_clean = loans.loc[~ex_mask].copy()

left_count = len(loans_clean); excl_count = len(excluded_df)
if excl_count > 0:
    st.warning(f"{excl_count} row(s) excluded due to opening_balance ≤ {min_ob:.2f}. These IDs will not be included in WA metrics, projections, or tiering.")
    st.download_button("⬇️ Download excluded rows (CSV)", data=excluded_df.to_csv(index=False).encode("utf-8"), file_name="excluded_zero_ob.csv", mime="text/csv")
else:
    st.info("No rows excluded.")

# Replace working dataset with cleaned one
loans = loans_clean

if len(loans) == 0:
    st.error("After applying the exclusion rule, no loans remain to model. Adjust the minimum opening balance or uncheck the exclusion.")
    st.stop()

# Auto set WA from tape if selected
if use_wa_from_tape:
    ob = loans["opening_balance"].astype(float)
    pd_w = (pd.to_numeric(loans["pd"], errors="coerce") * ob).sum() / ob.sum() if ob.sum()>0 else pd.to_numeric(loans["pd"], errors="coerce").mean()
    lgd_w = (pd.to_numeric(loans["lgd"], errors="coerce") * ob).sum() / ob.sum() if ob.sum()>0 else pd.to_numeric(loans["lgd"], errors="coerce").mean()
    if pd_w > 1: pd_w = pd_w / 100.0
    if lgd_w > 1: lgd_w = lgd_w / 100.0
    WA_PD_annual = pd_w
    WA_LGD = lgd_w
    st.info(f"WA PD set from tape: {WA_PD_annual*100:.2f}% | WA LGD set from tape: {WA_LGD*100:.2f}%")

# --- Run engine or reuse ---
if run_clicked or ("monthly_df" not in st.session_state):
    monthly_df, by_loan = project_cashflows(
        loans_df=loans, months=months, pd_annual=WA_PD_annual, lgd=WA_LGD,
        cpr_annual=CPR_annual, servicing_bps_pa=Servicing_bps, recovery_lag_months=Recovery_lag_m
    )
    st.session_state["monthly_df"] = monthly_df; st.session_state["by_loan"] = by_loan
    st.success("Projection complete.")

monthly_df = st.session_state["monthly_df"]
by_loan = st.session_state["by_loan"]

# === Scenario badge above charts ===
st.markdown(f"**Scenario in view:** `{sel_scn}`")

# --- KPIs ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Interest (horizon)", f"{monthly_df['interest_collected'].sum():,.0f} €")
c2.metric("Total Principal (horizon)", f"{(monthly_df['scheduled_principal']+monthly_df['prepayments']).sum():,.0f} €")
c3.metric("Defaults (horizon, before recov.)", f"{monthly_df['defaults'].sum():,.0f} €")
c4.metric("Recoveries (horizon)", f"{monthly_df['recoveries'].sum():,.0f} €")

st.subheader("📆 Portfolio monthly cash flows")
st.dataframe(monthly_df.set_index("month"), use_container_width=True)

# --- Portfolio outstanding balance chart (interactive) ---
st.subheader("📉 Portfolio outstanding balance over time")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=monthly_df["month"],
    y=monthly_df["ending_balance"],
    mode="lines",
    name="Ending balance",
    hovertemplate="Month %{x}<br>Outstanding €%{y:,.2f}<extra></extra>",
))
fig.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(
        title="Month",
        rangeslider=dict(visible=True),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
    ),
    yaxis=dict(
        title="Outstanding balance (€)",
        tickformat=",.2f",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        exponentformat="none",
        showexponent="none",
    ),
)
st.plotly_chart(fig, use_container_width=True)

# --- Cashflows by component (separate visualisations) ---
st.subheader("💧 Cashflows by component")

# Build component series
month_x = monthly_df["month"]
interest = monthly_df["interest_collected"]
sched_prin = monthly_df["scheduled_principal"]
prepay = monthly_df["prepayments"]
recoveries = monthly_df["recoveries"]
defaults = -monthly_df["defaults"]          # model as outflow (negative)
serv_fees = -monthly_df["servicing_fee"]    # outflow (negative)

# 1) Stacked monthly bars (relative mode: positives above, negatives below)
fig_stack = go.Figure()
fig_stack.add_trace(go.Bar(x=month_x, y=interest, name="Interest"))
fig_stack.add_trace(go.Bar(x=month_x, y=sched_prin, name="Scheduled principal"))
fig_stack.add_trace(go.Bar(x=month_x, y=prepay, name="Prepayments"))
fig_stack.add_trace(go.Bar(x=month_x, y=recoveries, name="Recoveries"))
fig_stack.add_trace(go.Bar(x=month_x, y=defaults, name="Defaults (outflow)"))
fig_stack.add_trace(go.Bar(x=month_x, y=serv_fees, name="Servicing fees (outflow)"))

fig_stack.update_layout(
    barmode="relative",
    template="plotly_dark",
    hovermode="x unified",
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(title="Month"),
    yaxis=dict(title="Monthly cashflow (€)", tickformat=",.0f"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_stack, use_container_width=True)

# 2) Cumulative lines (running totals by component)
cum_interest = interest.cumsum()
cum_prin = (sched_prin + prepay).cumsum()
cum_recov = recoveries.cumsum()
cum_defaults = (-defaults).cumsum()   # back to positive for display
cum_serv = (-serv_fees).cumsum()      # positive display
cum_net = (interest + sched_prin + prepay + recoveries + defaults + serv_fees).cumsum()

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(x=month_x, y=cum_interest, mode="lines", name="Interest (cum)"))
fig_cum.add_trace(go.Scatter(x=month_x, y=cum_prin, mode="lines", name="Principal + Prepay (cum)"))
fig_cum.add_trace(go.Scatter(x=month_x, y=cum_recov, mode="lines", name="Recoveries (cum)"))
fig_cum.add_trace(go.Scatter(x=month_x, y=cum_defaults, mode="lines", name="Defaults (cum)"))
fig_cum.add_trace(go.Scatter(x=month_x, y=cum_serv, mode="lines", name="Servicing fees (cum)"))
fig_cum.add_trace(go.Scatter(x=month_x, y=cum_net, mode="lines", name="Net cash to bank (cum)", line=dict(dash="dash")))

fig_cum.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(title="Month", rangeslider=dict(visible=True)),
    yaxis=dict(title="Cumulative cashflow (€)", tickformat=",.0f"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_cum, use_container_width=True)

with st.expander("🔍 Per-loan drilldown"):
    # Build a sorted list of loan_ids
    loan_ids = sorted(list(by_loan.keys()), key=lambda x: (len(str(x)), str(x)))
    sel = st.selectbox("Choose a loan_id", loan_ids if loan_ids else ["(none)"])
    if loan_ids:
        sched = by_loan[str(sel)]
        # Quick KPIs for the selected loan
        li1, li2, li3, li4 = st.columns(4)
        li1.metric("Opening balance (m1)", f"{(sched.loc[sched['month']==1,'opening_balance'].iloc[0] if 'opening_balance' in sched.columns and len(sched)>0 else 0):,.0f} €")
        li2.metric("Total interest (horizon)", f"{sched['interest'].sum():,.0f} €")
        li3.metric("Defaults − Recoveries", f"{(sched['default'].sum() - sched['recovery'].sum()):,.0f} €")
        li4.metric("Scheduled principal + Prepay", f"{(sched['scheduled_principal'].sum() + sched['prepayment'].sum()):,.0f} €")
        st.dataframe(sched, use_container_width=True, height=280)
        st.download_button("⬇️ Download this loan's schedule (CSV)",
                           data=sched.to_csv(index=False).encode("utf-8"),
                           file_name=f"loan_{sel}_schedule.csv",
                           mime="text/csv")
    else:
        st.info("No loans available for drilldown.")

# --- Valuation metrics ---
months_index = monthly_df["month"].astype(float)
principal_cash = monthly_df["scheduled_principal"] + monthly_df["prepayments"]
recoveries_cash = monthly_df["recoveries"]
r_m = (1 + disc_rate_annual) ** (1/12) - 1
def _disc(s): return sum(v / ((1 + r_m) ** int(m)) for m, v in zip(months_index, s))

npv = _npv(monthly_df["net_cash_to_bank"].values.tolist(), r_m)
wal_m = _wal(months_index, principal_cash)
wal_with_recov_m = _wal_with_recov(months_index, principal_cash, recoveries_cash)
defaults_12 = monthly_df.loc[monthly_df["month"] <= 12, "defaults"].sum()
recov_12 = monthly_df.loc[monthly_df["month"] <= 12, "recoveries"].sum()
ecl12 = defaults_12 - recov_12
ecl12_disc = _disc(monthly_df.loc[monthly_df["month"] <= 12, "defaults"]) - _disc(monthly_df.loc[monthly_df["month"] <= 12, "recoveries"])
lifetime_defaults = monthly_df["defaults"].sum()
lifetime_recov = monthly_df["recoveries"].sum()
lifetime_ecl = lifetime_defaults - lifetime_recov

pv_ratio = None
if 'opening_balance' in loans.columns:
    ob_total = float(loans['opening_balance'].sum())
    if ob_total > 0: pv_ratio = npv / ob_total

st.subheader("📌 Key valuation metrics")
cA, cB, cC, cD = st.columns(4)
cA.metric("PV of cash flows (at disc. rate)", f"{npv:,.0f} €")
cB.metric("WAL (principal-only, months)", f"{wal_m:,.1f}")
cC.metric("WAL (incl. recoveries, months)", f"{wal_with_recov_m:,.1f}")
cD.metric("PV / Opening Balance", f"{pv_ratio:.2f}x" if pv_ratio is not None else "N/A")

# --- Loan-level risk tiering & assignment ---
st.subheader("🧩 Loan-level risk tiering & assignment")
st.caption("Risk scoring uses modelled EL and your tape PD/LGD.")

# Build per-loan ECL metrics from schedules
per_loan_rows = []
for lid, sched in by_loan.items():
    el_h = float((sched["default"] - sched["recovery"]).sum())
    el12 = float(sched.loc[sched["month"] <= 12, "default"].sum() - sched.loc[sched["month"] <= 12, "recovery"].sum())
    ob0 = float(sched.loc[sched["month"] == 1, "opening_balance"].iloc[0]) if "opening_balance" in sched.columns and len(sched) else float("nan")
    per_loan_rows.append({"loan_id": lid, "EL_h": el_h, "EL12": el12, "OB": ob0, "ELpct_h": (el_h/ob0 if ob0 and ob0>0 else 0.0), "ELpct_12": (el12/ob0 if ob0 and ob0>0 else 0.0)})
risk_df = pd.DataFrame(per_loan_rows)

# Add PD/LGD and other fields from the cleaned tape (dtype-safe)
loans_join = loans[["loan_id","pd","lgd","opening_balance","remaining_term","monthly_rate"]].copy()
loans_join["loan_id"] = loans_join["loan_id"].astype(str)
risk_df["loan_id"] = risk_df["loan_id"].astype(str)
assign_df = risk_df.merge(loans_join, on="loan_id", how="left")

# Choose risk measure
measure = st.selectbox("Risk measure for assignment",
    options=["EL% over horizon", "EL amount over horizon", "12m EL%", "PD annual (from tape)", "PD×LGD (from tape)"], index=0)

if measure == "EL% over horizon":
    assign_df["score"] = assign_df["ELpct_h"].fillna(0.0); score_label = "EL%_h"
elif measure == "EL amount over horizon":
    assign_df["score"] = assign_df["EL_h"].fillna(0.0); score_label = "EL_amount_h"
elif measure == "12m EL%":
    assign_df["score"] = assign_df["ELpct_12"].fillna(0.0); score_label = "EL%_12m"
elif measure == "PD annual (from tape)":
    pd_vals = pd.to_numeric(assign_df["pd"], errors="coerce")
    if pd_vals.max() is not None and pd.notna(pd_vals.max()) and float(pd_vals.max()) > 1.0:
        pd_vals = pd_vals / 100.0
    assign_df["score"] = pd_vals.fillna(WA_PD_annual); score_label = "PD_annual"
else:
    pd_vals = pd.to_numeric(assign_df["pd"], errors="coerce")
    lgd_vals = pd.to_numeric(assign_df["lgd"], errors="coerce")
    if pd_vals.max() is not None and pd.notna(pd_vals.max()) and float(pd_vals.max()) > 1.0: pd_vals = pd_vals / 100.0
    if lgd_vals.max() is not None and pd.notna(lgd_vals.max()) and float(lgd_vals.max()) > 1.0: lgd_vals = lgd_vals / 100.0
    assign_df["score"] = (pd_vals.fillna(WA_PD_annual) * lgd_vals.fillna(WA_LGD)); score_label = "PDxLGD"

# Assignment method
method = st.radio("Assignment method", ["By thresholds", "Auto-quantiles"], horizontal=True)
def _assign_thresholds(df, low_thr, high_thr):
    bins = []
    for val in df["score"]:
        if val >= high_thr: bins.append("Equity")
        elif val <= low_thr: bins.append("Senior")
        else: bins.append("Mezzanine")
    return bins

if method == "By thresholds":
    if "PD" in score_label:
        low_thr  = st.number_input("Senior threshold (PD ≤, %)", 0.0, 100.0, 1.5, 0.1, format="%.2f")/100.0
        high_thr = st.number_input("Equity threshold (PD ≥, %)", 0.0, 100.0, 3.0, 0.1, format="%.2f")/100.0
    elif score_label == "PDxLGD":
        low_thr  = st.number_input("Senior threshold (PD×LGD ≤, %)", 0.0, 100.0, 1.0, 0.1, format="%.2f")/100.0
        high_thr = st.number_input("Equity threshold (PD×LGD ≥, %)", 0.0, 100.0, 2.0, 0.1, format="%.2f")/100.0
    else:
        low_thr  = st.number_input("Senior threshold (EL% ≤)", 0.0, 100.0, 0.5, 0.1, format="%.2f")/100.0
        high_thr = st.number_input("Equity threshold (EL% ≥)", 0.0, 100.0, 2.0, 0.1, format="%.2f")/100.0
    assign_df["Tranche"] = _assign_thresholds(assign_df, low_thr, high_thr)
else:
    eq_pct = st.number_input("Top % (highest risk) to Equity", 0.0, 100.0, 10.0, 1.0, format="%.0f")/100.0
    sr_pct = st.number_input("Bottom % (lowest risk) to Senior", 0.0, 100.0, 70.0, 1.0, format="%.0f")/100.0
    q_hi = assign_df["score"].quantile(1 - eq_pct) if eq_pct>0 else float("inf")
    q_lo = assign_df["score"].quantile(sr_pct) if sr_pct>0 else float("-inf")
    def _assign_quant(val):
        if val >= q_hi: return "Equity"
        elif val <= q_lo: return "Senior"
        else: return "Mezzanine"
    assign_df["Tranche"] = assign_df["score"].apply(_assign_quant)

# --- Ensure numeric opening balance for summary ---
if "opening_balance" in assign_df.columns:
    ob_num = pd.to_numeric(assign_df["opening_balance"], errors="coerce")
else:
    ob_num = pd.Series([float("nan")] * len(assign_df))
if ("OB" in assign_df.columns) and (ob_num.isna().all() or ob_num.sum() == 0):
    ob_num = pd.to_numeric(assign_df["OB"], errors="coerce")
assign_df["OB_used"] = ob_num.fillna(0.0)

# Summary by tranche
def _fmt_pct_safe(x):
    import math
    if pd.isna(x) or (isinstance(x, float) and (x==float("inf") or x==float("-inf"))):
        return "-"
    return f"{x*100:.2f}%"

sum_tbl = assign_df.groupby("Tranche", dropna=False).agg(
    **{"Number of loans": ("Tranche","size")},
    OB_total=("OB_used","sum"),
    EL_h_total=("EL_h","sum"),
).reset_index()
sum_tbl["WA_ELpct_h"] = sum_tbl.apply(lambda r: (r["EL_h_total"]/r["OB_total"]) if r["OB_total"]>0 else float("nan"), axis=1)
sum_tbl = sum_tbl.sort_values("Tranche", na_position="last")

st.subheader("Assignment summary")
st.dataframe(sum_tbl.style.format({"OB_total":"{:,.0f}", "EL_h_total":"{:,.0f}", "WA_ELpct_h": _fmt_pct_safe}), use_container_width=True)

# Show lists by tranche — hide columns that appear in the user input file (except loan_id),
# cast loan_id to integer, and set it as the index to remove the extra index column.
input_cols = set(loans.columns)                 # columns from the uploaded/cleaned user tape
cols_to_exclude = input_cols.difference({"loan_id"})  # everything from the tape except loan_id

# Keep modelling outputs (EL_h, ELpct_h, score, Tranche, etc.), but drop tape columns
display_cols_all = [c for c in assign_df.columns if (c not in cols_to_exclude) or (c == "loan_id")]

for tr in ["Equity", "Mezzanine", "Senior"]:
    with st.expander(f"{tr} - loan list"):
        view = assign_df.loc[assign_df["Tranche"] == tr, display_cols_all].copy()

        # Ensure loan_id is integer and make it the index (removes default 0..N index col)
        if "loan_id" in view.columns:
            view["loan_id"] = pd.to_numeric(view["loan_id"], errors="coerce").astype("Int64")
            view = view.set_index("loan_id")

        # Sort: Seniors ascending risk (score), Equity/Mezz descending (higher risk first)
        order_asc = (tr == "Senior")
        if "score" in view.columns:
            view = view.sort_values("score", ascending=order_asc)

        st.dataframe(view, use_container_width=True)

# Downloads for assignment
out_assign = assign_df.copy()
st.download_button("⬇️ Download loan assignment (CSV)", data=out_assign.to_csv(index=False).encode("utf-8"), file_name="loan_assignment.csv", mime="text/csv")

xbuf = BytesIO()
with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
    sum_tbl.to_excel(writer, index=False, sheet_name="Summary")
    assign_df.to_excel(writer, index=False, sheet_name="Assignment")
    for tr in ["Equity","Mezzanine","Senior"]:
        tr_df = assign_df.loc[assign_df["Tranche"]==tr]
        tr_df.to_excel(writer, index=False, sheet_name=tr)
st.download_button("⬇️ Download assignment pack (Excel)", data=xbuf.getvalue(), file_name="srt_assignment_pack.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- SRT Tranching (prototype) ---
st.subheader("🏦 SRT Tranching")
st.caption("Simple loss waterfall on expected losses (defaults − recoveries) within horizon. Equity absorbs first losses, then Mezzanine, then Senior.")

total_ob = float(loans["opening_balance"].sum()) if "opening_balance" in loans.columns else None
if total_ob is None or total_ob <= 0:
    st.warning("Opening balance column not found or non-positive; cannot run tranching.")
else:
    c1, c2, c3 = st.columns(3)
    eq_pct = c1.number_input("Equity thickness (% of OB)", 0.0, 100.0, 5.0, 0.5, format="%.1f")/100.0
    mez_pct = c2.number_input("Mezzanine thickness (% of OB)", 0.0, 100.0, 10.0, 0.5, format="%.1f")/100.0
    sr_pct = max(0.0, 1.0 - (eq_pct + mez_pct))
    c3.metric("Senior thickness (% of OB)", f"{sr_pct*100:.1f}%")

    loss_series = (monthly_df["defaults"] - monthly_df["recoveries"]).clip(lower=0.0).tolist()

    eq_notional = total_ob * eq_pct
    mez_notional = total_ob * mez_pct
    sr_notional = total_ob * sr_pct

    eq_loss = mez_loss = sr_loss = 0.0
    eq_rem, mez_rem, sr_rem = eq_notional, mez_notional, sr_notional

    alloc = []
    for m, L in enumerate(loss_series, start=1):
        l = float(L)
        take_eq = min(l, eq_rem); eq_rem -= take_eq; l -= take_eq
        take_mez = min(l, mez_rem); mez_rem -= take_mez; l -= take_mez
        take_sr = min(l, sr_rem); sr_rem -= take_sr; l -= take_sr
        eq_loss += take_eq; mez_loss += take_mez; sr_loss += take_sr
        alloc.append({"month": m, "eq_loss": take_eq, "mez_loss": take_mez, "sr_loss": take_sr, "eq_rem": eq_rem, "mez_rem": mez_rem, "sr_rem": sr_rem})

    alloc_df = pd.DataFrame(alloc)
    sum_df = pd.DataFrame([
        {"Tranche": "Equity", "Original Notional": eq_notional, "Loss Allocated": eq_loss, "Loss % of Tranche": (eq_loss/eq_notional*100 if eq_notional>0 else np.nan), "Remain Notional": eq_rem},
        {"Tranche": "Mezzanine", "Original Notional": mez_notional, "Loss Allocated": mez_loss, "Loss % of Tranche": (mez_loss/mez_notional*100 if mez_notional>0 else np.nan), "Remain Notional": mez_rem},
        {"Tranche": "Senior", "Original Notional": sr_notional, "Loss Allocated": sr_loss, "Loss % of Tranche": (sr_loss/sr_notional*100 if sr_notional>0 else np.nan), "Remain Notional": sr_rem},
    ])

    total_el = float((monthly_df["defaults"] - monthly_df["recoveries"]).clip(lower=0.0).sum())
    transferred_el = eq_loss + mez_loss
    st.metric("Indicative EL transferred (Equity+Mezz / Portfolio EL)", f"{(transferred_el/total_el*100 if total_el>0 else 0):.1f}%")

    st.subheader("Tranche summary")
    st.dataframe(sum_df.style.format({"Original Notional": "{:,.0f}", "Loss Allocated": "{:,.0f}", "Loss % of Tranche": "{:,.1f}%", "Remain Notional": "{:,.0f}"}), use_container_width=True)

    with st.expander("Monthly loss allocation"):
        st.dataframe(alloc_df, use_container_width=True)


# --- ECL Analysis Section ---
from engine import compute_ecl

st.header("🧮 IFRS 9 ECL Analysis")

run_ecl = st.sidebar.checkbox("Run ECL Analysis", value=False)
if run_ecl and file is not None:
    try:
        loan_tape = pd.read_excel(file, sheet_name="loan_tape")
        assumptions_df = pd.read_excel(file, sheet_name="assumptions")
        assumptions_dict = {str(row[0]).strip(): row[1] for _, row in assumptions_df.iterrows() if str(row[0]).strip()}

        loan_level, month_level = compute_ecl(
            loans_df=loan_tape,
            assumptions=assumptions_dict,
            horizon_months=months,
            discount_rate_annual=disc_rate_annual,
            use_loan_eir=True
        )

        totals = loan_level[['ecl_12m','ecl_lifetime']].sum()
        col1, col2 = st.columns(2)
        col1.metric("12-month ECL", f"{totals['ecl_12m']:,.0f}")
        col2.metric("Lifetime ECL", f"{totals['ecl_lifetime']:,.0f}")

        st.subheader("Loan-level ECL")
        st.dataframe(loan_level)

        st.download_button("⬇️ Download Loan-level ECL", loan_level.to_csv(index=False), "ecl_loan_level.csv", "text/csv")
        st.download_button("⬇️ Download Month-level ECL", month_level.to_csv(index=False), "ecl_month_level.csv", "text/csv")
    except Exception as e:
        st.error(f"ECL computation failed: {e}")


# --- Basel Capital Impact ---
st.header("🏦 Basel Capital Impact")
from engine import compute_rwa_capital

with st.expander("Configure capital approach"):
    approach = st.selectbox(
        "Approach",
        ["IRB-Corporate", "IRB-ResidentialMortgage", "IRB-OtherRetail", "Standardised-Flat"],
        index=0
    )
    pillar1_ratio = st.number_input("Capital ratio (Pillar 1)", min_value=0.01, max_value=0.20, value=0.08, step=0.01, help="Commonly 8% under Basel")
    std_rw = st.number_input("Standardised flat RW% (if selected)", min_value=0.0, max_value=1250.0, value=100.0, step=5.0)

if file is not None and st.sidebar.checkbox("Run Basel Capital Impact", value=False):
    try:
        loan_tape = pd.read_excel(file, sheet_name="loan_tape")
        # Ensure remaining_term exists
        if 'remaining_term' not in loan_tape.columns and 'maturity_months' in loan_tape.columns and 'months_elapsed' in loan_tape.columns:
            loan_tape['remaining_term'] = (loan_tape['maturity_months'] - loan_tape['months_elapsed']).clip(lower=1)

        rwa_loan, rwa_totals = compute_rwa_capital(
            loans_df=loan_tape,
            approach=approach,
            ead_col="opening_balance",
            pd_col="pd",
            lgd_col="lgd",
            remaining_term_col="remaining_term",
            std_risk_weight_pct=std_rw,
            capital_ratio=pillar1_ratio
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("EAD (total)", f"{rwa_totals['EAD']:,.0f}")
        c2.metric("RWA (total)", f"{rwa_totals['RWA']:,.0f}")
        c3.metric(f"Capital @ {pillar1_ratio*100:.1f}%", f"{rwa_totals['capital_requirement']:,.0f}")

        st.subheader("Per-loan RWA and Capital")
        st.dataframe(rwa_loan)

        st.download_button("⬇️ Download RWA per loan", rwa_loan.to_csv(index=False), "rwa_per_loan.csv", "text/csv")
    except Exception as e:
        st.error(f"Basel capital computation failed: {e}")


# --- FAQ Chatbot Section (dropdown version) ---
st.subheader("💬 FAQ Chatbot")

faq = {
    "What is this app for?": "It forecasts cashflows for a loan portfolio and helps analyze risks.",
    "What input file do I need?": "Upload an Excel file with a sheet named 'loan_tape' containing the required columns.",
    "What is PD?": "Probability of Default (annualised), i.e., the chance a loan defaults within a year.",
    "What is LGD?": "Loss Given Default – the percentage of exposure lost after a default and recovery.",
    "What is CPR?": "Conditional Prepayment Rate, the annualized rate at which borrowers prepay their loans.",
    "What is WAL?": "Weighted Average Life, the average time until principal is repaid.",
    "What is NPV?": "Net Present Value, the discounted value of future net cash flows using your chosen discount rate.",
    "What are recoveries?": "Cash inflows from defaulted loans, collected after the recovery lag and adjusted by LGD.",
    "What are servicing fees?": "Ongoing fees deducted each month, calculated as basis points (bps) on the opening balance.",
    "What is tranche assignment?": "It allocates loans into Equity, Mezzanine, or Senior risk buckets based on expected losses or PD/LGD metrics.",
    "What is SRT tranching?": "Significant Risk Transfer tranching simulates how portfolio losses are allocated through a loss waterfall across Equity, Mezzanine, and Senior.",
    "What is ECL?": "Expected Credit Loss, calculated as Defaults minus Recoveries either over 12 months or full horizon.",
    "What does PV / Opening Balance mean?": "It shows the ratio of the present value of projected cash flows to the portfolio’s starting balance, like a valuation multiple."
}

selected_q = st.selectbox("Choose a question:", ["(select a question)"] + list(faq.keys()))

if selected_q != "(select a question)":
    st.success(faq[selected_q])

# --- FAQ Table with Search + Downloads ---
with st.expander("📖 Show all FAQs"):
    import pandas as pd
    from io import BytesIO

    faq_df = pd.DataFrame(list(faq.items()), columns=["Question", "Answer"])

    # Search filter
    search_term = st.text_input("🔍 Search FAQs")
    if search_term:
        mask = faq_df.apply(
            lambda row: search_term.lower() in row["Question"].lower() or search_term.lower() in row["Answer"].lower(),
            axis=1
        )
        filtered_df = faq_df[mask]
    else:
        filtered_df = faq_df

    st.dataframe(filtered_df, use_container_width=True)

    # Download buttons
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download FAQs as CSV", data=csv_data, file_name="faqs.csv", mime="text/csv")

    xlsx_buf = BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
        filtered_df.to_excel(writer, index=False, sheet_name="FAQs")
    st.download_button("⬇️ Download FAQs as Excel", data=xlsx_buf.getvalue(), file_name="faqs.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
