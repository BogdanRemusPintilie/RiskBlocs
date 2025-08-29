
import pandas as pd
import numpy as np

def pmt(rate, nper, pv):
    if rate == 0:
        return pv / max(nper, 1)
    return pv * (rate * (1 + rate) ** nper) / ((1 + rate) ** nper - 1)

def annual_to_monthly_pd(pd_annual):
    return 1 - (1 - pd_annual) ** (1/12)

def cpr_to_smm(cpr):
    return 1 - (1 - cpr) ** (1/12)

def project_cashflows(loans_df, months=60, pd_annual=0.025, lgd=0.725, cpr_annual=0.22,
                      servicing_bps_pa=100, recovery_lag_months=12):
    """
    loans_df must have at least: loan_id, opening_balance, remaining_term, monthly_rate, monthly_payment
    """
    df = loans_df.copy()
    required = ["loan_id","opening_balance","remaining_term","monthly_rate","monthly_payment"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    pd_m = annual_to_monthly_pd(pd_annual)
    smm = cpr_to_smm(cpr_annual)
    serv_m = (servicing_bps_pa / 10000.0) / 12.0

    agg = []
    per_loan = {str(row["loan_id"]): [] for _, row in df.iterrows()}

    state = []
    for _, row in df.iterrows():
        state.append({
            'loan_id': str(row["loan_id"]),
            'bal': float(row["opening_balance"]),
            'rate': float(row["monthly_rate"]),
            'pmt': float(row["monthly_payment"]),
            'mrem': int(round(row["remaining_term"])),
            'rec_sched': {}
        })

    for t in range(1, months+1):
        tot_sched_prin = tot_int = tot_prepay = tot_default = tot_recovery = tot_serv = 0.0

        for s in state:
            if s['bal'] <= 1e-8 or s['mrem'] <= 0:
                per_loan[s['loan_id']].append({
                    'month': t, 'opening_balance': 0.0, 'interest': 0.0,
                    'scheduled_principal': 0.0, 'prepayment': 0.0, 'default': 0.0,
                    'recovery': 0.0, 'servicing_fee': 0.0, 'ending_balance': 0.0
                })
                continue

            ob = s['bal']
            r = s['rate']
            pmt = min(s['pmt'], ob*(1+r)+1e-9)

            interest = ob * r
            sched_prin = max(0.0, pmt - interest)
            sched_prin = min(sched_prin, ob)

            bal_after_sched = ob - sched_prin
            exp_default = min(bal_after_sched * pd_m, bal_after_sched)
            bal_after_default = bal_after_sched - exp_default

            exp_prepay = min(bal_after_default * smm, bal_after_default)
            eb = bal_after_default - exp_prepay

            serv_fee = ob * serv_m

            fut = t + int(recovery_lag_months)
            if exp_default > 0:
                s['rec_sched'][fut] = s['rec_sched'].get(fut, 0.0) + exp_default * (1 - lgd)
            rec = s['rec_sched'].pop(t, 0.0)

            per_loan[s['loan_id']].append({
                'month': t,
                'opening_balance': ob,
                'interest': interest,
                'scheduled_principal': sched_prin,
                'prepayment': exp_prepay,
                'default': exp_default,
                'recovery': rec,
                'servicing_fee': serv_fee,
                'ending_balance': eb
            })

            tot_sched_prin += sched_prin
            tot_int += interest
            tot_prepay += exp_prepay
            tot_default += exp_default
            tot_recovery += rec
            tot_serv += serv_fee

            s['bal'] = eb
            s['mrem'] = max(0, s['mrem'] - 1)

        agg.append({
            'month': t,
            'interest_collected': tot_int,
            'scheduled_principal': tot_sched_prin,
            'prepayments': tot_prepay,
            'defaults': tot_default,
            'recoveries': tot_recovery,
            'servicing_fee': tot_serv,
            'net_cash_to_bank': tot_int + tot_sched_prin + tot_prepay + tot_recovery - tot_serv,
            'ending_balance': sum(s['bal'] for s in state)
        })

    monthly_df = pd.DataFrame(agg)
    loan_schedules = {k: pd.DataFrame(v) for k, v in per_loan.items()}
    return monthly_df, loan_schedules


# =========================
# IFRS 9 ECL EXTENSION
# =========================
def compute_ecl(
    loans_df: pd.DataFrame,
    assumptions: dict = None,
    horizon_months: int | None = None,
    discount_rate_annual: float | None = None,
    use_loan_eir: bool = True,
    stop_when_fully_amortized: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute IFRS 9-style expected credit losses at loan level.

    This function calculates both 12-month ECL and lifetime ECL using a
    monthly hazard approximation:

        ECL = sum_t [ Survival(t-1) * PD_m(t) * LGD * EAD(t) * DF(t) ]

    where EAD(t) is the expected outstanding at the *start* of month t
    on a path with scheduled amortization and prepayments (no defaults).

    Parameters
    ----------
    loans_df : DataFrame
        Must include: 
            - 'loan_id', 'opening_balance', 'interest_rate (annual)',
              'maturity_months', 'months_elapsed', 'remaining_term',
              'monthly_rate', 'monthly_payment', 'pd', 'lgd'
        If some fields are missing, reasonable defaults are inferred.
    assumptions : dict, optional
        Can include 'SMM' (single monthly mortality for prepayments).
        If not provided, SMM defaults to 0.0.
    horizon_months : int, optional
        Max number of months to consider. If None, uses each loan's
        'remaining_term'.
    discount_rate_annual : float, optional
        Annual discount rate used for present value. If None and
        use_loan_eir=True, uses each loan's own EIR from
        'interest_rate (annual)'; otherwise 0.
    use_loan_eir : bool
        Whether to default to each loan's EIR when discount_rate_annual
        is not given.
    stop_when_fully_amortized : bool
        If True, stops the EAD path for a loan when balance <= 0.

    Returns
    -------
    loan_level_ecl : DataFrame
        Columns: loan_id, ecl_12m, ecl_lifetime, ead_start, rem_term, lgd,
                 pd_month, smm, discount_rate_annual_used
    month_level_ecl : DataFrame
        Per-loan, per-month contributions with columns:
        [loan_id, month, ead, mdp, survival, df, el_contrib_12m, el_contrib_life]
    """
    df = loans_df.copy()
    required_cols = ['loan_id', 'opening_balance']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Defaults and fallbacks
    if 'interest_rate (annual)' not in df.columns and 'monthly_rate' in df.columns:
        df['interest_rate (annual)'] = (df['monthly_rate'].fillna(0) * 12).clip(lower=0)
    if 'monthly_rate' not in df.columns and 'interest_rate (annual)' in df.columns:
        df['monthly_rate'] = (df['interest_rate (annual)'].fillna(0) / 12).clip(lower=0)
    if 'monthly_payment' not in df.columns:
        # fall back to standard amortization payment using remaining_term
        # protect against missing remaining_term
        if 'remaining_term' not in df.columns:
            df['remaining_term'] = (df.get('maturity_months', pd.Series([60]*len(df))).fillna(60)
                                    - df.get('months_elapsed', pd.Series([0]*len(df))).fillna(0)).clip(lower=1)
        df['monthly_payment'] = df.apply(lambda r: pmt(r['monthly_rate'], int(max(1, round(r['remaining_term']))), r['opening_balance']), axis=1)
    if 'remaining_term' not in df.columns:
        df['remaining_term'] = (df.get('maturity_months', pd.Series([60]*len(df))).fillna(60)
                                - df.get('months_elapsed', pd.Series([0]*len(df))).fillna(0)).clip(lower=1)
    if 'pd' not in df.columns:
        # try to use annual WA PD if present in assumptions
        pd_m = 0.0
        if assumptions and 'WA_PD_annual' in assumptions:
            pd_m = annual_to_monthly_pd(float(assumptions['WA_PD_annual']))
        df['pd'] = pd_m
    if 'lgd' not in df.columns:
        df['lgd'] = float(assumptions.get('WA_LGD', 0.45)) if assumptions else 0.45

    smm = 0.0
    if assumptions and 'SMM' in assumptions:
        smm = float(assumptions['SMM'])
    elif assumptions and 'CPR_annual_base' in assumptions:
        smm = cpr_to_smm(float(assumptions['CPR_annual_base']))

    # Determine discounting approach
    if discount_rate_annual is None and use_loan_eir:
        df['_disc_rate_annual'] = df['interest_rate (annual)'].fillna(0.0)
    else:
        df['_disc_rate_annual'] = float(discount_rate_annual or 0.0)
    df['_disc_rate_monthly'] = df['_disc_rate_annual'] / 12.0

    # Containers for outputs
    rows_loan = []
    rows_month = []

    for _, r in df.iterrows():
        loan_id = r['loan_id']
        bal = float(r['opening_balance'])
        mrate = float(r.get('monthly_rate', 0.0))
        pay = float(r.get('monthly_payment', 0.0))
        rem = int(max(1, round(r.get('remaining_term', 1))))
        pd_m = float(r.get('pd', 0.0))
        lgd = float(r.get('lgd', 0.45))
        dr_m = float(r.get('_disc_rate_monthly', 0.0))

        T = rem if horizon_months is None else min(rem, int(horizon_months))
        survival = 1.0
        ecl_life = 0.0
        ecl_12 = 0.0
        month = 0

        # Keep the starting EAD for reporting
        ead_start = bal

        while month < T and (bal > 1e-8 or not stop_when_fully_amortized):
            month += 1
            # EAD at the start of month
            ead = bal

            # marginal default probability at month t
            mdp = survival * pd_m

            # discount factor (end of month discounting)
            df_t = 1.0 / ((1.0 + dr_m) ** month)

            el = mdp * lgd * ead * df_t
            ecl_life += el
            if month <= 12:
                ecl_12 += el

            rows_month.append({
                'loan_id': loan_id,
                'month': month,
                'ead': ead,
                'mdp': mdp,
                'survival': survival,
                'df': df_t,
                'el_contrib': el,
                'is_12m_window': month <= 12
            })

            # Update survival for next month (no default occurred yet)
            survival *= (1.0 - pd_m)

            # Amortization path without defaults (scheduled + prepayment)
            interest = bal * mrate
            sched_prin = max(0.0, min(bal, pay - interest))
            bal = bal - sched_prin
            prepay = smm * bal
            bal = max(0.0, bal - prepay)

            if stop_when_fully_amortized and bal <= 1e-8:
                bal = 0.0
                break

        rows_loan.append({
            'loan_id': loan_id,
            'ecl_12m': ecl_12,
            'ecl_lifetime': ecl_life,
            'ead_start': ead_start,
            'rem_term': rem,
            'lgd': lgd,
            'pd_month': pd_m,
            'smm': smm,
            'discount_rate_annual_used': dr_m * 12.0
        })

    loan_level = pd.DataFrame(rows_loan)
    month_level = pd.DataFrame(rows_month)
    return loan_level, month_level

# =========================
# Basel Capital Impact (IRB & Standardised)
# =========================
import math
import pandas as pd

def _norm_cdf(x):
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327 * math.exp(-x * x / 2.0)
    prob = 1.0 - d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + 1.330274429 * t))))
    return prob if x >= 0 else 1.0 - prob

def _inv_norm_cdf(p):
    if p <= 0 or p >= 1:
        return float('nan')
    a1, a2, a3, a4, a5, a6 = -39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924
    b1, b2, b3, b4, b5 = -54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857
    c1, c2, c3, c4, c5, c6 = -0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878
    d1, d2, d3, d4 = 0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = (-2 * math.log(p)) ** 0.5
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    if phigh < p:
        q = (-2 * math.log(1 - p)) ** 0.5
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)

def _clip_pd(pdval):
    return max(0.0003, min(0.99999, float(pdval)))

def _mat_years_from_rem_term(rem_months):
    return max(1.0, min(5.0, float(rem_months) / 12.0))

def _IRB_corporate_K(pdval, lgd, M):
    pdval = _clip_pd(pdval)
    R = 0.12 * (1 - math.exp(-50 * pdval)) / (1 - math.exp(-50)) + 0.24 * (1 - (1 - math.exp(-50 * pdval)) / (1 - math.exp(-50)))
    b = (0.08451 - 0.05898 * math.log(pdval)) ** 2
    Gpd = _inv_norm_cdf(pdval)
    G999 = _inv_norm_cdf(0.999)
    K = lgd * _norm_cdf((1.0 / (1 - R) ** 0.5) * Gpd + ((R / (1 - R)) ** 0.5) * G999)
    K -= pdval * lgd
    K *= (1.0 + (M - 2.5) * b) / (1.0 - 1.5 * b)
    return max(0.0, K)

def _IRB_resi_mortgage_K(pdval, lgd):
    pdval = _clip_pd(pdval)
    R = 0.15
    Gpd = _inv_norm_cdf(pdval)
    G999 = _inv_norm_cdf(0.999)
    K = lgd * _norm_cdf((1.0 / (1 - R) ** 0.5) * Gpd + ((R / (1 - R)) ** 0.5) * G999)
    return max(0.0, K)

def _IRB_other_retail_K(pdval, lgd):
    pdval = _clip_pd(pdval)
    R = 0.02 * (1 - math.exp(-35 * pdval)) / (1 - math.exp(-35)) + 0.17 * (1 - (1 - math.exp(-35 * pdval)) / (1 - math.exp(-35)))
    Gpd = _inv_norm_cdf(pdval)
    G999 = _inv_norm_cdf(0.999)
    K = lgd * _norm_cdf((1.0 / (1 - R) ** 0.5) * Gpd + ((R / (1 - R)) ** 0.5) * G999)
    return max(0.0, K)

def compute_rwa_capital(
    loans_df,
    approach = "IRB-Corporate",
    ead_col = "opening_balance",
    pd_col = "pd",
    lgd_col = "lgd",
    remaining_term_col = "remaining_term",
    std_risk_weight_pct = 75.0,
    class_col = None,
    class_to_rw = None,
    capital_ratio = 0.08
):
    df = loans_df.copy()
    if 'loan_id' not in df.columns:
        df['loan_id'] = range(1, len(df) + 1)
    if ead_col not in df.columns:
        raise ValueError("EAD column '%s' not found" % ead_col)

    rows = []
    for _, r in df.iterrows():
        loan_id = r['loan_id']
        EAD = float(r[ead_col])
        if EAD <= 0:
            rows.append({'loan_id': loan_id, 'EAD': EAD, 'RWA': 0.0, 'capital_requirement': 0.0, 'K': 0.0})
            continue

        if approach == "IRB-Corporate":
            if pd_col not in df.columns or lgd_col not in df.columns:
                raise ValueError("IRB approaches require PD and LGD columns")
            pd_v = float(r[pd_col])
            lgd_v = float(r[lgd_col])
            M = _mat_years_from_rem_term(r.get(remaining_term_col, 60.0))
            K = _IRB_corporate_K(pd_v, lgd_v, M)
            RWA = 12.5 * K * EAD
            cap = capital_ratio * RWA
            rows.append({'loan_id': loan_id, 'EAD': EAD, 'K': K, 'RWA': RWA, 'capital_requirement': cap, 'approach':'IRB-Corporate'})
        elif approach == "IRB-ResidentialMortgage":
            if pd_col not in df.columns or lgd_col not in df.columns:
                raise ValueError("IRB approaches require PD and LGD columns")
            pd_v = float(r[pd_col])
            lgd_v = float(r[lgd_col])
            K = _IRB_resi_mortgage_K(pd_v, lgd_v)
            RWA = 12.5 * K * EAD
            cap = capital_ratio * RWA
            rows.append({'loan_id': loan_id, 'EAD': EAD, 'K': K, 'RWA': RWA, 'capital_requirement': cap, 'approach':'IRB-ResidentialMortgage'})
        elif approach == "IRB-OtherRetail":
            if pd_col not in df.columns or lgd_col not in df.columns:
                raise ValueError("IRB approaches require PD and LGD columns")
            pd_v = float(r[pd_col])
            lgd_v = float(r[lgd_col])
            K = _IRB_other_retail_K(pd_v, lgd_v)
            RWA = 12.5 * K * EAD
            cap = capital_ratio * RWA
            rows.append({'loan_id': loan_id, 'EAD': EAD, 'K': K, 'RWA': RWA, 'capital_requirement': cap, 'approach':'IRB-OtherRetail'})
        elif approach == "Standardised-Flat":
            rw_pct = float(std_risk_weight_pct) / 100.0
            RWA = rw_pct * EAD
            cap = capital_ratio * RWA
            rows.append({'loan_id': loan_id, 'EAD': EAD, 'risk_weight_pct': std_risk_weight_pct, 'RWA': RWA, 'capital_requirement': cap, 'approach':'Standardised-Flat'})
        elif approach == "Standardised-Mapping":
            if class_col is None or class_to_rw is None:
                raise ValueError("Standardised-Mapping requires class_col and class_to_rw")
            cls = str(r.get(class_col, "Unknown"))
            rw_pct = float(class_to_rw.get(cls, std_risk_weight_pct)) / 100.0
            RWA = rw_pct * EAD
            cap = capital_ratio * RWA
            rows.append({'loan_id': loan_id, 'EAD': EAD, 'class': cls, 'risk_weight_pct': rw_pct*100, 'RWA': RWA, 'capital_requirement': cap, 'approach':'Standardised-Mapping'})
        else:
            raise ValueError("Unknown approach '%s'" % approach)

    out = pd.DataFrame(rows)
    totals = out[['EAD','RWA','capital_requirement']].sum()
    return out, totals
