import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import calendar
import warnings
warnings.filterwarnings('ignore')

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    LIFETIMES_AVAILABLE = True
except ImportError:
    LIFETIMES_AVAILABLE = False

st.set_page_config(page_title="Retail Dashboard", page_icon="📈", layout="wide")

st.markdown("""
<style>
    h1, h2, h3 { color: #1f2937; font-weight: 700; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #0066cc;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .metric-title {
        font-size: 11px;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 6px;
    }
    .metric-comparison { font-size: 12px; color: #6b7280; }
    .metric-comp-value { font-weight: 600; color: #374151; }
    .metric-delta-pos { color: #059669; font-weight: 700; }
    .metric-delta-neg { color: #dc2626; font-weight: 700; }
    .filter-row { background: #f8f9fa; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

FILE_PATH = 'challenge1_retail_churn - Datasource.xlsm'

# ============================================================================
# MINIMAL FIX: Dynamic Forecast Title + Period-based Calculations
# ============================================================================
HORIZON_LABELS = {
    '3 Months': (3, 'Quarterly'),
    '6 Months': (6, 'Half-Annual'),
    '12 Months': (12, 'Yearly')
}

def get_forecast_report_title(horizon_label):
    """Convert horizon to period name for report title."""
    months, period_name = HORIZON_LABELS.get(horizon_label, (3, 'Quarterly'))
    return f"{period_name} Forecast vs Actuals"

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_data():
    trans = pd.read_excel(FILE_PATH, sheet_name='Transactions')
    trans.columns = trans.columns.str.strip()
    trans = trans[['transaction_id', 'customer_id', 'transaction_date', 'category', 'amount', 'store_region']]
    trans['transaction_date'] = pd.to_datetime(trans['transaction_date'])
    trans['amount'] = (
        trans['amount'].astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )

    raw = pd.read_excel(FILE_PATH, sheet_name='RFM Template', header=None)
    header_row = None
    for i, row in raw.iterrows():
        if row.astype(str).str.contains('Ranking', case=False, na=False).any():
            header_row = i
            break
    rfm_ref = pd.read_excel(FILE_PATH, sheet_name='RFM Template',
                             header=header_row if header_row is not None else 0)
    rfm_ref.columns = rfm_ref.columns.str.strip()
    rfm_ref = rfm_ref.dropna(how='all')

    return trans, rfm_ref


# ============================================================================
# RFM FUNCTIONS
# ============================================================================
def score_series(series, ascending=True):
    n = len(series)
    if n <= 1:
        return pd.Series([5] * n, index=series.index)
    rank = series.rank(method='min', ascending=True)
    pct = (rank - 1) / (n - 1)
    if not ascending:
        pct = 1 - pct
    return np.ceil(np.minimum(5, np.maximum(1, pct * 5))).astype(int)


@st.cache_data(show_spinner=False)
def calculate_rfm_as_of_date(transactions, cutoff_date):
    filtered_trans = transactions[transactions['transaction_date'] <= cutoff_date].copy()
    if len(filtered_trans) == 0:
        return pd.DataFrame()

    all_customers = filtered_trans.groupby('customer_id').agg(
        Last_Transaction_Date=('transaction_date', 'max'),
        First_Transaction_Date=('transaction_date', 'min'),
        Frequency=('transaction_date', 'count'),
        Monetary=('amount', 'sum')
    ).reset_index()

    is_new = (
        (all_customers['First_Transaction_Date'].dt.year == cutoff_date.year) &
        (all_customers['First_Transaction_Date'].dt.month == cutoff_date.month)
    )
    new_customers = all_customers[is_new].copy()
    existing_customers = all_customers[~is_new].copy()

    new_customers['Recency'] = (cutoff_date - new_customers['Last_Transaction_Date']).dt.days
    new_customers['R_Score'] = np.nan
    new_customers['F_Score'] = np.nan
    new_customers['M_Score'] = np.nan
    new_customers['Segment'] = 'New Customer'

    if len(existing_customers) > 0:
        existing_customers['Recency'] = (cutoff_date - existing_customers['Last_Transaction_Date']).dt.days
        existing_customers['R_Score'] = score_series(existing_customers['Recency'], ascending=False)
        existing_customers['F_Score'] = score_series(existing_customers['Frequency'], ascending=True)
        existing_customers['M_Score'] = score_series(existing_customers['Monetary'], ascending=True)
        existing_customers['Segment'] = None

    return pd.concat([existing_customers, new_customers], ignore_index=True)


def parse_pattern(pattern_str):
    """Parse a 'R≥4,F≥4,M≥4' style string into a list of (key, op, value) conditions."""
    if pd.isna(pattern_str) or str(pattern_str).strip() == '':
        return []
    conditions = []
    for part in str(pattern_str).split(','):
        part = part.strip()
        if not part:
            continue
        if '≤' in part:
            key, val = part.split('≤')
            conditions.append((key.strip(), '≤', int(val.strip())))
        elif '≥' in part:
            key, val = part.split('≥')
            conditions.append((key.strip(), '≥', int(val.strip())))
        elif '=' in part and '-' in part.split('=')[1]:
            key, val = part.split('=')
            lo, hi = val.split('-')
            conditions.append((key.strip(), '=range', (int(lo.strip()), int(hi.strip()))))
        elif '=' in part:
            key, val = part.split('=')
            conditions.append((key.strip(), '=', int(val.strip())))
        elif '<' in part:
            key, val = part.split('<')
            conditions.append((key.strip(), '<', int(val.strip())))
        elif '>' in part:
            key, val = part.split('>')
            conditions.append((key.strip(), '>', int(val.strip())))
    return conditions


def build_condition_mask(rfm_df, conditions):
    """
    Vectorized equivalent of evaluating one pattern's conditions across the
    WHOLE dataframe at once, instead of per-customer. Returns a boolean
    Series aligned to rfm_df.index.
    """
    if not conditions:
        return pd.Series(False, index=rfm_df.index)
    col_map = {'R': 'R_Score', 'F': 'F_Score', 'M': 'M_Score'}
    mask = pd.Series(True, index=rfm_df.index)
    for key, op, val in conditions:
        col = col_map.get(key)
        if col is None or col not in rfm_df.columns:
            return pd.Series(False, index=rfm_df.index)
        series = rfm_df[col]
        if op == '≤':
            cond = series <= val
        elif op == '≥':
            cond = series >= val
        elif op == '=':
            cond = series == val
        elif op == '=range':
            cond = series.between(val[0], val[1])
        elif op == '<':
            cond = series < val
        elif op == '>':
            cond = series > val
        else:
            cond = pd.Series(False, index=rfm_df.index)
        mask &= cond
    return mask


@st.cache_data(show_spinner=False)
def assign_segments(rfm_df, rfm_ref_table):
    """
    Assign each customer to the first matching RFM segment (by Ranking order),
    vectorized across all customers at once rather than looping row-by-row.

    Same semantics as the original implementation: for each row missing a
    Segment, walk the pattern rows in Ranking order and take the first whose
    conditions all hold against that row's R/F/M scores.
    """
    rfm_df = rfm_df.copy()
    segment_colors = {}
    for _, row in rfm_ref_table.iterrows():
        cat = row.get('Category')
        if pd.notna(cat):
            bg = row.get('Background Color', '#3b82f6')
            fg = row.get('Front Color', '#ffffff')
            segment_colors[str(cat)] = {'bg': str(bg), 'fg': str(fg)}

    pattern_rows = rfm_ref_table[
        rfm_ref_table['RFM Pattern'].notna() &
        rfm_ref_table['RFM Pattern'].astype(str).str.strip().ne('') &
        rfm_ref_table['Ranking'].notna()
    ].copy()
    pattern_rows['Ranking'] = pd.to_numeric(pattern_rows['Ranking'], errors='coerce')
    pattern_rows = pattern_rows.dropna(subset=['Ranking']).sort_values('Ranking')
    # Parse each pattern string once (was: re-parsed per customer per pattern row)
    pattern_rows['_conditions'] = pattern_rows['RFM Pattern'].apply(parse_pattern)

    needs_assignment = rfm_df['Segment'].isna()
    if needs_assignment.any():
        sub = rfm_df.loc[needs_assignment]
        condlist, choicelist = [], []
        for _, ref_row in pattern_rows.iterrows():
            condlist.append(build_condition_mask(sub, ref_row['_conditions']).values)
            choicelist.append(str(ref_row['Category']))

        if condlist:
            assigned = np.select(condlist, choicelist, default=None)
            rfm_df.loc[needs_assignment, 'Segment'] = assigned

    rfm_df.loc[rfm_df['Segment'].isna(), 'Segment'] = 'Unclassified'
    return rfm_df, segment_colors


# ============================================================================
# RFM SNAPSHOT CACHING (Forecast tab historical trend)
# ============================================================================
@st.cache_data(show_spinner=False)
def get_historical_segment_revenue(
    transactions: pd.DataFrame,
    rfm_ref: pd.DataFrame,
    hist_start: pd.Timestamp,
    forecast_start: pd.Timestamp,
    lookback_months: int
):
    """
    Pre-compute cumulative revenue-by-segment for each historical month-end,
    in a single cached call instead of one cache lookup per month.

    Mirrors the original per-month loop exactly: for each month-end, take all
    transactions up to that date, tag them with the RFM segment each customer
    held as of that date, and sum amount by segment.

    Returns:
        hist_months_axis: list of month label strings (e.g. 'Jan 2026'), in order
        hist_cumulative_data: dict of {month_label: {segment: cumulative_revenue}}
    """
    hist_months_axis = []
    hist_cumulative_data = {}

    for i in range(lookback_months):
        month_end = (hist_start + pd.DateOffset(months=i + 1)).replace(day=1) + pd.DateOffset(days=-1)
        if month_end >= forecast_start:
            break
        month_str = month_end.strftime('%b %Y')
        hist_months_axis.append(month_str)

        trans_to_date = transactions[transactions['transaction_date'] <= month_end]
        rfm_at_date = calculate_rfm_as_of_date(transactions, month_end)
        rfm_at_date, _ = assign_segments(rfm_at_date, rfm_ref)
        trans_to_date = trans_to_date.merge(rfm_at_date[['customer_id', 'Segment']], on='customer_id', how='left')
        trans_to_date['Segment'] = trans_to_date['Segment'].fillna('Unknown')
        hist_cumulative_data[month_str] = trans_to_date.groupby('Segment')['amount'].sum().to_dict()

    return hist_months_axis, hist_cumulative_data


# ============================================================================
# MONTHLY TRENDS HELPERS (Revenue, Transactions, Active Customers)
# ============================================================================
@st.cache_data(show_spinner=False)
def calculate_monthly_metrics(
    transactions: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    regions=None,
    categories=None
):
    """
    Calculate monthly revenue, transaction count, and active customer count.
    Returns: DataFrame with columns [Month, Revenue, Transactions, Active_Customers]
    """
    trans = transactions.copy()
    
    # Apply filters
    if regions:
        trans = trans[trans['store_region'].isin(regions)]
    if categories:
        trans = trans[trans['category'].isin(categories)]
    
    # Filter by date range
    trans = trans[
        (trans['transaction_date'] >= start_date) &
        (trans['transaction_date'] <= end_date)
    ].copy()
    
    if len(trans) == 0:
        return pd.DataFrame(columns=['Month', 'Revenue', 'Transactions', 'Active_Customers'])
    
    # Create month column
    trans['Month'] = trans['transaction_date'].dt.to_period('M')
    
    # Aggregate by month
    monthly = trans.groupby('Month').agg(
        Revenue=('amount', 'sum'),
        Transactions=('transaction_id', 'count'),
        Active_Customers=('customer_id', 'nunique')
    ).reset_index()
    
    monthly['Month'] = monthly['Month'].astype(str)
    
    return monthly


# ============================================================================
# SEGMENT ANALYSIS HELPERS
# ============================================================================
@st.cache_data(show_spinner=False)
def get_segment_monthly_trends(transactions, rfm_ref, eom_date, lookback_months=12, regions=None):
    """
    Calculate monthly revenue and customer count by segment for the last N months,
    anchored to eom_date (the selected Reference End of Month).
    Returns: (months_axis, revenue_by_segment_dict, count_by_segment_dict)
    where each dict maps segment_name -> [list of monthly values]
    """
    if regions:
        transactions = transactions[transactions['store_region'].isin(regions)]
    
    months_axis = []
    revenue_trends = {}
    count_trends = {}
    
    end_date = pd.Timestamp(eom_date)
    start_date = end_date - pd.DateOffset(months=lookback_months-1)
    
    for i in range(lookback_months):
        month_end = (start_date + pd.DateOffset(months=i+1)).replace(day=1) + pd.DateOffset(days=-1)
        month_end = month_end.replace(day=min(month_end.day, 28))  # Avoid day overflow
        month_str = month_end.strftime('%b %y')
        months_axis.append(month_str)
        
        trans_month = transactions[transactions['transaction_date'] <= month_end]
        rfm_month = calculate_rfm_as_of_date(transactions, month_end)
        
        if len(rfm_month) > 0:
            rfm_month, _ = assign_segments(rfm_month, rfm_ref)
            trans_month_seg = trans_month.merge(rfm_month[['customer_id', 'Segment']], on='customer_id', how='left')
            trans_month_seg['Segment'] = trans_month_seg['Segment'].fillna('Unknown')
            
            seg_revenue = trans_month_seg.groupby('Segment')['amount'].sum()
            seg_count = trans_month_seg.groupby('Segment')['customer_id'].nunique()
            
            for seg in seg_revenue.index:
                if seg not in revenue_trends:
                    revenue_trends[seg] = []
                    count_trends[seg] = []
                revenue_trends[seg].append(seg_revenue[seg])
                count_trends[seg].append(seg_count[seg])
    
    # Pad missing segments with zeros
    all_segments = set(revenue_trends.keys())
    for seg in all_segments:
        if len(revenue_trends[seg]) < lookback_months:
            revenue_trends[seg] = [0] * (lookback_months - len(revenue_trends[seg])) + revenue_trends[seg]
            count_trends[seg] = [0] * (lookback_months - len(count_trends[seg])) + count_trends[seg]
    
    return months_axis, revenue_trends, count_trends


def get_segment_retention_curves(transactions, rfm_ref, segments_list, cutoff_date, lookback_months=12, regions=None):
    """
    Build retention curves for cohorts acquired within the lookback window, by segment.

    For each segment:
      - Identifies which customers belong to that segment as of cutoff_date
      - Restricts to customers first acquired within [cohort_start, cutoff_date]
      - Computes milestone retention rates (M3, M6, M9, M12) only for milestones
        where sufficient data exists (target month <= cutoff_date)

    Returns:
        retention_by_segment: dict {segment: {milestone_label: retention_rate}}
        valid_milestone_labels: list of milestones with data (e.g. ['M3','M6','M9'])
    """
    if regions:
        transactions = transactions[transactions['store_region'].isin(regions)]
    
    cutoff_date = pd.Timestamp(cutoff_date)
    cohort_start = cutoff_date - pd.DateOffset(months=lookback_months - 1)

    # Build cohort mechanics from raw transactions, capped at cutoff_date
    df = transactions[transactions['transaction_date'] <= cutoff_date].copy()
    df['txn_month'] = df['transaction_date'].values.astype('datetime64[M]')

    first_txn = df.groupby('customer_id')['transaction_date'].min()
    cohort_month = first_txn.values.astype('datetime64[M]')
    cohort_map = pd.Series(cohort_month, index=first_txn.index)
    df['cohort_month'] = df['customer_id'].map(cohort_map)
    df['months_since'] = (
        (df['txn_month'].dt.year  - df['cohort_month'].dt.year)  * 12 +
        (df['txn_month'].dt.month - df['cohort_month'].dt.month)
    )

    # Customers first acquired within the analysis window
    window_customers = set(cohort_map[
        (cohort_map >= cohort_start.to_datetime64()) &
        (cohort_map <= cutoff_date.to_datetime64())
    ].index)

    # Segment membership as of cutoff_date
    rfm_at_cutoff = calculate_rfm_as_of_date(transactions, cutoff_date)
    rfm_at_cutoff, _ = assign_segments(rfm_at_cutoff, rfm_ref)
    segment_lookup = rfm_at_cutoff.set_index('customer_id')['Segment']

    # Only include milestones reachable within the window
    milestones_all = [3, 6, 9, 12]
    max_possible_months = (
        (cutoff_date.year  - pd.Timestamp(cohort_start).year)  * 12 +
        (cutoff_date.month - pd.Timestamp(cohort_start).month)
    )
    valid_milestones = [m for m in milestones_all if m <= max_possible_months]

    retention_by_segment = {}

    for segment in segments_list:
        if segment in ('Unknown', 'New Customer'):
            continue

        seg_customers = set(segment_lookup[segment_lookup == segment].index)
        seg_window = window_customers & seg_customers

        if not seg_window:
            continue

        seg_df = df[df['customer_id'].isin(seg_window)]
        m0_count = seg_df[seg_df['months_since'] == 0]['customer_id'].nunique()

        if m0_count == 0:
            continue

        seg_retention = {}
        for m in valid_milestones:
            active = seg_df[seg_df['months_since'] == m]['customer_id'].nunique()
            seg_retention[f'M{m}'] = active / m0_count

        retention_by_segment[segment] = seg_retention

    valid_milestone_labels = [f'M{m}' for m in valid_milestones]
    return retention_by_segment, valid_milestone_labels


def get_segment_migration_matrix(transactions, rfm_ref, start_date, end_date, regions=None):
    """
    Build a migration matrix showing customer movement between segments
    from period start to period end.
    Returns:
        migration_matrix: DataFrame rows=start_segment, cols=end_segment, values=customer count
        counts_start: Series of customer counts per segment at start_date
        counts_end: Series of customer counts per segment at end_date
    """
    if regions:
        transactions = transactions[transactions['store_region'].isin(regions)]
    
    rfm_start = calculate_rfm_as_of_date(transactions, start_date)
    rfm_end = calculate_rfm_as_of_date(transactions, end_date)

    if len(rfm_start) == 0 or len(rfm_end) == 0:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=int)

    rfm_start, _ = assign_segments(rfm_start, rfm_ref)
    rfm_end, _ = assign_segments(rfm_end, rfm_ref)

    counts_start = rfm_start['Segment'].value_counts()
    counts_end   = rfm_end['Segment'].value_counts()

    # New customers: present at end but not at start
    existing_ids = set(rfm_start['customer_id'])
    rfm_new = rfm_end[~rfm_end['customer_id'].isin(existing_ids)]
    counts_new = rfm_new['Segment'].value_counts()

    migration = rfm_start[['customer_id', 'Segment']].rename(columns={'Segment': 'Segment_Start'})
    migration = migration.merge(
        rfm_end[['customer_id', 'Segment']].rename(columns={'Segment': 'Segment_End'}),
        on='customer_id', how='inner'
    )

    migration_matrix = pd.crosstab(
        migration['Segment_Start'],
        migration['Segment_End'],
        margins=False
    )

    return migration_matrix, counts_start, counts_end, counts_new


def get_segment_health_scorecard(transactions, rfm_ref, cutoff_date, lookback_months=12, regions=None):
    """
    Calculate health metrics per segment as of cutoff_date:
    - Growth rate (% change in customer count over lookback window)
    - Avg CLV (if lifetimes available)
    - Revenue contribution (%)
    
    Growth % compares customer count at cutoff_date vs. cutoff_date - lookback_months
    """
    if regions:
        transactions = transactions[transactions['store_region'].isin(regions)]
    
    comparison_date = cutoff_date - pd.DateOffset(months=lookback_months)
    
    # Current period RFM
    rfm_curr = calculate_rfm_as_of_date(transactions, cutoff_date)
    if len(rfm_curr) == 0:
        return pd.DataFrame()
    
    rfm_curr, _ = assign_segments(rfm_curr, rfm_ref)
    
    # Comparison period RFM
    rfm_prev = calculate_rfm_as_of_date(transactions, comparison_date)
    if len(rfm_prev) > 0:
        rfm_prev, _ = assign_segments(rfm_prev, rfm_ref)
    else:
        rfm_prev = pd.DataFrame()
    
    scorecard_data = []
    
    for segment in sorted(rfm_curr['Segment'].unique()):
        if pd.isna(segment) or segment == 'Unknown':
            continue
        
        curr_count = len(rfm_curr[rfm_curr['Segment'] == segment])
        prev_count = len(rfm_prev[rfm_prev['Segment'] == segment]) if len(rfm_prev) > 0 else 0
        
        growth = ((curr_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
        
        curr_revenue = rfm_curr[rfm_curr['Segment'] == segment]['Monetary'].sum()
        total_revenue = rfm_curr['Monetary'].sum()
        revenue_pct = (curr_revenue / total_revenue * 100) if total_revenue > 0 else 0
        
        curr_recency = rfm_curr[rfm_curr['Segment'] == segment]['Recency'].mean()
        
        scorecard_data.append({
            'Segment': segment,
            'Customers': curr_count,
            'Growth %': growth,
            'Avg Revenue €': curr_revenue / curr_count if curr_count > 0 else 0,
            'Revenue %': revenue_pct,
            'Avg Recency Days': curr_recency
        })
    
    return pd.DataFrame(scorecard_data).sort_values('Revenue %', ascending=False)


# ============================================================================
# PERIOD HELPERS
# ============================================================================
def end_of_month(dt):
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    return pd.Timestamp(dt.year, dt.month, last_day)


def get_period_dates(period_name, eom_date):
    eom = pd.Timestamp(eom_date)
    if period_name == 'CM':
        return eom.replace(day=1), eom
    elif period_name == 'This QTR':
        q_start_month = ((eom.month - 1) // 3) * 3 + 1
        return eom.replace(month=q_start_month, day=1), eom
    elif period_name == 'Last QTR':
        q_start_month = ((eom.month - 1) // 3) * 3 + 1
        end = eom.replace(month=q_start_month, day=1) - timedelta(days=1)
        start = (end - pd.DateOffset(months=2)).replace(day=1)
        return pd.Timestamp(start), end_of_month(end)
    elif period_name == 'YTD':
        return eom.replace(month=1, day=1), eom
    elif period_name == 'LTM':
        start = (eom - pd.DateOffset(months=11)).replace(day=1)
        return pd.Timestamp(start), eom
    elif period_name == 'LY':
        return pd.Timestamp(eom.year - 1, 1, 1), pd.Timestamp(eom.year - 1, 12, 31)
    return eom, eom


def get_comparison_dates(period_name, curr_start, curr_end, comparison_type):
    if period_name == 'LY':
        return pd.Timestamp(curr_start.year - 1, 1, 1), pd.Timestamp(curr_start.year - 1, 12, 31)
    if comparison_type == 'Previous Year':
        comp_start = curr_start - pd.DateOffset(years=1)
        comp_end = curr_end - pd.DateOffset(years=1)
        return pd.Timestamp(comp_start), end_of_month(comp_end)
    # Previous Period
    if period_name == 'CM':
        comp_end = curr_start - timedelta(days=1)
        return comp_end.replace(day=1), end_of_month(comp_end)
    elif period_name == 'This QTR':
        # Mirror the same month positions in the previous quarter
        # e.g. Oct only → Jul only, Oct–Nov → Jul–Aug, Oct–Nov–Dec → Jul–Sep
        comp_start = curr_start - pd.DateOffset(months=3)
        comp_end = end_of_month(curr_end - pd.DateOffset(months=3))
        return pd.Timestamp(comp_start), pd.Timestamp(comp_end)
    elif period_name == 'Last QTR':
        comp_end = curr_start - timedelta(days=1)
        comp_start = (comp_end - pd.DateOffset(months=2)).replace(day=1)
        return pd.Timestamp(comp_start), end_of_month(comp_end)
    elif period_name == 'YTD':
        comp_end = (curr_end.replace(day=1) - timedelta(days=1))
        return curr_start, end_of_month(comp_end)
    elif period_name == 'LTM':
        # Shift the entire 12-month window back by 1 month
        # e.g. current 01.08.2021→31.07.2022 → comparison 01.07.2021→30.06.2022
        comp_start = (curr_start - pd.DateOffset(months=1))
        comp_end = end_of_month(curr_end - pd.DateOffset(months=1))
        return pd.Timestamp(comp_start), pd.Timestamp(comp_end)
    return curr_start, curr_end


def metric_card(title, curr_val, comp_val=None, fmt='currency'):
    if fmt == 'currency':
        curr_str = f"€{curr_val:,.0f}"
    elif fmt == 'integer':
        curr_str = f"{int(curr_val):,}"
    elif fmt == 'percentage':
        curr_str = f"{curr_val:.1f}%"
    else:
        curr_str = f"{curr_val:,.1f}"

    if comp_val is None:
        # Single-value card: title + value only, no comparison row
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{curr_str}</div>
        </div>
        """, unsafe_allow_html=True)
        return

    if fmt == 'currency':
        comp_str = f"€{comp_val:,.0f}"
    elif fmt == 'integer':
        comp_str = f"{int(comp_val):,}"
    elif fmt == 'percentage':
        comp_str = f"{comp_val:.1f}%"
    else:
        comp_str = f"{comp_val:,.1f}"

    if comp_val and comp_val != 0:
        if fmt == 'percentage':
            delta = curr_val - comp_val  # percentage points
            delta_str = f"{delta:+.1f}pp"
        else:
            delta = (curr_val - comp_val) / abs(comp_val) * 100
            delta_str = f"{delta:+.1f}%"
        delta_class = "metric-delta-pos" if delta >= 0 else "metric-delta-neg"
    else:
        delta_str = "N/A"
        delta_class = "metric-comp-value"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{curr_str}</div>
        <div class="metric-comparison">
            vs <span class="metric-comp-value">{comp_str}</span>
            &nbsp;&nbsp;<span class="{delta_class}">{delta_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def calculate_customer_cohorts(transactions, curr_start, curr_end, comp_start, comp_end,
                               pre_curr_start, pre_curr_end, pre_comp_start, pre_comp_end,
                               regions=None, categories=None):
    """Calculate new customers, churned customers, and rates for current and comparison periods.
    
    Respects region and category filters and period-aware logic.
    - New Customers: first_transaction_date within period
    - Churned Customers: active in pre-period but inactive in current period
    """
    
    def filter_trans(start, end):
        trans = transactions[
            (transactions['transaction_date'] >= start) &
            (transactions['transaction_date'] <= end)
        ]
        if regions:
            trans = trans[trans['store_region'].isin(regions)]
        if categories:
            trans = trans[trans['category'].isin(categories)]
        return trans
    
    # Calculate first transaction date for each customer (respecting filters)
    filtered_trans = transactions
    if regions:
        filtered_trans = filtered_trans[filtered_trans['store_region'].isin(regions)]
    if categories:
        filtered_trans = filtered_trans[filtered_trans['category'].isin(categories)]
    
    first_trans_date = filtered_trans.groupby('customer_id')['transaction_date'].min()
    
    # Current period data
    trans_curr = filter_trans(curr_start, curr_end)
    active_curr = set(trans_curr['customer_id'].unique())
    
    # Comparison period data
    trans_comp = filter_trans(comp_start, comp_end)
    active_comp = set(trans_comp['customer_id'].unique())
    
    # Pre-current period data
    trans_pre_curr = filter_trans(pre_curr_start, pre_curr_end)
    active_pre_curr = set(trans_pre_curr['customer_id'].unique())
    
    # Pre-comparison period data
    trans_pre_comp = filter_trans(pre_comp_start, pre_comp_end)
    active_pre_comp = set(trans_pre_comp['customer_id'].unique())
    
    # NEW CUSTOMERS: first transaction within the period (vectorized boolean mask + sum,
    # instead of a per-customer Python loop with repeated Series indexing)
    new_curr = int(((first_trans_date >= curr_start) & (first_trans_date <= curr_end)).sum())
    new_comp = int(((first_trans_date >= comp_start) & (first_trans_date <= comp_end)).sum())
    
    # CHURNED CUSTOMERS: active in pre-period but not in current period
    churned_curr = len(active_pre_curr - active_curr)
    churned_comp = len(active_pre_comp - active_comp)
    
    # ACQUISITION RATE: new / total active in current period
    acq_rate_curr = (new_curr / len(active_curr) * 100) if len(active_curr) > 0 else 0
    acq_rate_comp = (new_comp / len(active_comp) * 100) if len(active_comp) > 0 else 0
    
    # CHURN RATE: churned / active in pre-period
    churn_rate_curr = (churned_curr / len(active_pre_curr) * 100) if len(active_pre_curr) > 0 else 0
    churn_rate_comp = (churned_comp / len(active_pre_comp) * 100) if len(active_pre_comp) > 0 else 0
    
    return {
        'new_curr': new_curr, 'new_comp': new_comp,
        'churned_curr': churned_curr, 'churned_comp': churned_comp,
        'acq_rate_curr': acq_rate_curr, 'acq_rate_comp': acq_rate_comp,
        'churn_rate_curr': churn_rate_curr, 'churn_rate_comp': churn_rate_comp
    }


def bar_chart(x, y, colors, title='', height=350, fmt='currency'):
    total = sum(y) if sum(y) else 1
    if fmt == 'currency':
        text = [f"€{v/1000:.0f}k<br>{v/total*100:.1f}%" for v in y]
    else:
        text = [f"{v:,.0f}<br>{v/total*100:.1f}%" for v in y]
    fig = go.Figure([go.Bar(
        x=x, y=y, marker_color=colors, text=text, textposition='outside',
        hovertemplate='<b>%{x}</b><br>%{y:,.0f}<extra></extra>'
    )])
    fig.update_layout(
        title=title, height=height, showlegend=False,
        margin=dict(t=60 if title else 40, b=20, l=20, r=20),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_showgrid=False, yaxis_showgrid=True, yaxis_gridcolor='#e5e7eb'
    )
    fig.update_traces(cliponaxis=False)
    return fig


# ============================================================================
# COHORT RETENTION ANALYSIS
# ============================================================================
def build_monthly_cohort_table(transactions, value='customers', max_months_back=12, milestones=(0, 3, 6, 9, 12)):
    """
    Build a cohort progression table from transaction history.

    Rows: acquisition cohort month (the month of each customer's first transaction)
    Columns: milestone months since acquisition (0, 3, 6, 9, 12)
    Values: either active customer count or revenue, for customers from that
            cohort who transacted in (cohort_month + milestone)

    Only the last `max_months_back` complete cohorts (relative to the data's
    max transaction date) are returned, since older/incomplete cohorts on
    the right edge can't reach later milestones yet.
    """
    df = transactions.copy()
    df['txn_month'] = df['transaction_date'].values.astype('datetime64[M]')

    first_txn = df.groupby('customer_id')['transaction_date'].min()
    cohort_month = first_txn.values.astype('datetime64[M]')
    cohort_map = pd.Series(cohort_month, index=first_txn.index)
    df['cohort_month'] = df['customer_id'].map(cohort_map)

    df['months_since'] = (
        (df['txn_month'].dt.year - df['cohort_month'].dt.year) * 12 +
        (df['txn_month'].dt.month - df['cohort_month'].dt.month)
    )

    max_data_month = df['txn_month'].max()
    all_cohorts = sorted(df['cohort_month'].unique())
    # Keep only cohorts old enough to have a real (non-future) value for milestone 0,
    # and cap to the requested lookback window
    eligible_cohorts = [c for c in all_cohorts if c <= max_data_month]
    eligible_cohorts = eligible_cohorts[-max_months_back:]

    rows = []
    cohort_size = {}
    for c in eligible_cohorts:
        cohort_customers = cohort_map[cohort_map == c].index
        cohort_size[c] = len(cohort_customers)
        row = {'Cohort': pd.Timestamp(c).strftime('%b %Y'), '_cohort_date': pd.Timestamp(c), 'New Customers': len(cohort_customers)}
        for m in milestones:
            target_month = pd.Timestamp(c) + pd.DateOffset(months=m)
            if target_month > max_data_month:
                row[f'M{m}'] = np.nan  # milestone hasn't happened yet in the data
                continue
            slice_df = df[(df['cohort_month'] == c) & (df['months_since'] == m)]
            if value == 'customers':
                row[f'M{m}'] = slice_df['customer_id'].nunique()
            else:
                row[f'M{m}'] = slice_df['amount'].sum()
        rows.append(row)

    return pd.DataFrame(rows), cohort_size


def calculate_retention_medians(transactions, max_months_back=12, milestones=(0, 3, 6, 9, 12)):
    """
    Calculate, for each milestone month, the median retention rate
    (active customers at month M / customers acquired at month 0)
    across the last `max_months_back` cohorts that have reached that milestone.

    Returns a dict {milestone: median_retention_rate} plus the underlying
    cohort table (counts) for display.
    """
    cohort_df, cohort_size = build_monthly_cohort_table(
        transactions, value='customers', max_months_back=max_months_back, milestones=milestones
    )

    retention_medians = {}
    for m in milestones:
        col = f'M{m}'
        if col not in cohort_df.columns:
            continue
        valid = cohort_df.dropna(subset=[col])
        if len(valid) == 0 or m == 0:
            retention_medians[m] = 1.0 if m == 0 else np.nan
            continue
        rates = valid[col] / valid['New Customers'].replace(0, np.nan)
        retention_medians[m] = rates.median()

    return retention_medians, cohort_df


def calculate_acquisition_and_revenue_medians(transactions, lookback_months=12):
    """
    Median monthly new-customer acquisition count and median revenue per new
    customer (in their acquisition month), based on the last `lookback_months`
    complete months of data.
    """
    df = transactions.copy()
    df['txn_month'] = df['transaction_date'].values.astype('datetime64[M]')
    first_txn = df.groupby('customer_id')['transaction_date'].min()
    first_txn_month = first_txn.values.astype('datetime64[M]')
    cohort_map = pd.Series(first_txn_month, index=first_txn.index)

    max_month = df['txn_month'].max()
    months_window = [pd.Timestamp(max_month) - pd.DateOffset(months=i) for i in range(lookback_months)]
    months_window = pd.to_datetime(months_window)

    monthly_new = []
    monthly_rev_per_new = []
    for m in months_window:
        cust_in_month = cohort_map[cohort_map == m].index
        n_new = len(cust_in_month)
        if n_new == 0:
            continue
        rev = df[(df['customer_id'].isin(cust_in_month)) & (df['txn_month'] == m)]['amount'].sum()
        monthly_new.append(n_new)
        monthly_rev_per_new.append(rev / n_new)

    median_acquisition = float(np.median(monthly_new)) if monthly_new else 0.0
    median_revenue_per_new = float(np.median(monthly_rev_per_new)) if monthly_rev_per_new else 0.0
    return median_acquisition, median_revenue_per_new


def calculate_cohort_assumption_ranges(transactions, max_months_back=12, milestones=(0, 3, 6, 9, 12)):
    """
    Compute the min/max range observed across the last `max_months_back`
    acquisition cohorts for: new-customer acquisition count, average revenue
    per new customer (at M0), and retention rate at each milestone.

    These min/max values are the basis for the forecast sliders (so slider
    bounds reflect actual historical variation rather than a fixed +/-% of
    the median).
    """
    cohort_customers_df, _ = build_monthly_cohort_table(
        transactions, value='customers', max_months_back=max_months_back, milestones=milestones
    )
    cohort_revenue_df, _ = build_monthly_cohort_table(
        transactions, value='revenue', max_months_back=max_months_back, milestones=milestones
    )

    # Acquisition count range: New Customers column, all rows (every cohort has M0 by definition)
    acquisition_counts = cohort_customers_df['New Customers']
    acquisition_min = float(acquisition_counts.min()) if len(acquisition_counts) else 0.0
    acquisition_max = float(acquisition_counts.max()) if len(acquisition_counts) else 0.0
    acquisition_median = float(acquisition_counts.median()) if len(acquisition_counts) else 0.0

    # Revenue per new customer range: M0 revenue / New Customers, per cohort
    valid_rev = cohort_revenue_df.dropna(subset=['M0'])
    rev_per_new = (valid_rev['M0'] / valid_rev['New Customers'].replace(0, np.nan)).dropna()
    revenue_min = float(rev_per_new.min()) if len(rev_per_new) else 0.0
    revenue_max = float(rev_per_new.max()) if len(rev_per_new) else 0.0
    revenue_median = float(rev_per_new.median()) if len(rev_per_new) else 0.0

    # Retention rate range per milestone: M{n} customers / New Customers, per cohort
    retention_ranges = {}
    retention_medians = {}
    for m in milestones:
        col = f'M{m}'
        if col not in cohort_customers_df.columns:
            continue
        if m == 0:
            retention_ranges[m] = (1.0, 1.0)
            retention_medians[m] = 1.0
            continue
        valid = cohort_customers_df.dropna(subset=[col])
        if len(valid) == 0:
            retention_ranges[m] = (np.nan, np.nan)
            retention_medians[m] = np.nan
            continue
        rates = valid[col] / valid['New Customers'].replace(0, np.nan)
        retention_ranges[m] = (float(rates.min()), float(rates.max()))
        retention_medians[m] = float(rates.median())

    return {
        'acquisition_min': acquisition_min, 'acquisition_max': acquisition_max, 'acquisition_median': acquisition_median,
        'revenue_min': revenue_min, 'revenue_max': revenue_max, 'revenue_median': revenue_median,
        'retention_ranges': retention_ranges, 'retention_medians': retention_medians
    }


# ============================================================================
# EXISTING CUSTOMER FORECAST (BG/NBD + GAMMA-GAMMA)
# ============================================================================
@st.cache_resource
def fit_bgnbd_gammagamma(transactions):
    """
    Fit BG/NBD (purchase frequency) and Gamma-Gamma (monetary value) models
    on the full transaction history, as of the max transaction date.
    Returns the fitted models plus the per-customer summary frame, or
    (None, None, None) if the lifetimes package isn't installed.
    """
    if not LIFETIMES_AVAILABLE:
        return None, None, None

    df = transactions.copy()
    observation_end = df['transaction_date'].max()

    summary = df.groupby('customer_id').agg(
        min_date=('transaction_date', 'min'),
        max_date=('transaction_date', 'max'),
        frequency=('transaction_date', 'count'),
        monetary_value=('amount', 'mean')
    ).reset_index()

    summary['T'] = (observation_end - summary['min_date']).dt.days
    summary['recency'] = (summary['max_date'] - summary['min_date']).dt.days
    # BG/NBD frequency = repeat purchases only (count - 1)
    summary['frequency'] = (summary['frequency'] - 1).clip(lower=0)

    fitter_input = summary[summary['T'] > 0].copy()

    bgf = None
    for penalizer in (0.01, 0.1, 0.5, 1.0, 2.0):
        try:
            bgf = BetaGeoFitter(penalizer_coef=penalizer)
            bgf.fit(fitter_input['frequency'], fitter_input['recency'], fitter_input['T'])
            break
        except Exception:
            bgf = None
            continue
    if bgf is None:
        return None, None, None

    ggf_input = fitter_input[(fitter_input['frequency'] > 0) & (fitter_input['monetary_value'] > 0)]
    ggf = None
    for penalizer in (0.01, 0.1, 0.5, 1.0, 2.0):
        try:
            ggf = GammaGammaFitter(penalizer_coef=penalizer)
            ggf.fit(ggf_input['frequency'], ggf_input['monetary_value'])
            break
        except Exception:
            ggf = None
            continue
    if ggf is None:
        return None, None, None

    return bgf, ggf, fitter_input


def forecast_existing_customer_clv(bgf, ggf, fitter_input, months=12):
    """
    Project existing-customer CLV over the given horizon (months) using the
    fitted BG/NBD + Gamma-Gamma models. Returns total predicted revenue.
    """
    if bgf is None or ggf is None or fitter_input is None or len(fitter_input) == 0:
        return 0.0

    clv_input = fitter_input[fitter_input['monetary_value'] > 0].copy()
    if len(clv_input) == 0:
        return 0.0

    clv = ggf.customer_lifetime_value(
        bgf,
        clv_input['frequency'],
        clv_input['recency'],
        clv_input['T'],
        clv_input['monetary_value'],
        time=months,
        freq='D'
    )
    return float(clv.sum())


# ============================================================================
# NEW CUSTOMER FORECAST (ACQUISITION x RETENTION x REVENUE, WITH CASCADE)
# ============================================================================
def build_new_customer_forecast(forecast_months, acquisition_per_month, revenue_per_new_customer,
                                 retention_medians, start_date):
    """
    Build a forward-looking cohort table for new customers acquired over the
    forecast horizon, applying the retention cascade at each milestone.

    Returns a DataFrame: one row per forecast acquisition month, with
    columns M0, M3, M6, M9, M12 showing retained customer counts (capped to
    the forecast horizon), plus a 'Revenue' column (total projected revenue
    for that cohort across the horizon).
    """
    milestones = sorted(retention_medians.keys())
    rows = []

    for i in range(forecast_months):
        cohort_date = pd.Timestamp(start_date) + pd.DateOffset(months=i)
        row = {'Cohort': cohort_date.strftime('%b %Y'), 'New Customers': round(acquisition_per_month)}
        cohort_revenue = round(acquisition_per_month) * revenue_per_new_customer  # M0 revenue

        for m in milestones:
            months_remaining_in_horizon = forecast_months - i - 1
            if m == 0:
                row['M0'] = round(acquisition_per_month)
            elif m <= months_remaining_in_horizon:
                retained = acquisition_per_month * retention_medians.get(m, 0)
                row[f'M{m}'] = round(retained)
                cohort_revenue += retained * revenue_per_new_customer
            else:
                row[f'M{m}'] = np.nan  # falls outside the forecast horizon

        row['Revenue'] = cohort_revenue
        rows.append(row)

    return pd.DataFrame(rows)


def style_cohort_heatmap(df, value_cols, fmt='{:,.0f}', na_rep='—', extra_formats=None):
    """
    Apply a green (high) -> yellow -> red (low) background gradient across
    value columns. NaN cells (milestone not yet reached) are forced to a
    plain white background instead of being colored by the gradient.

    extra_formats: optional dict of {column_name: format_string} for columns
    outside value_cols (e.g. 'Revenue') that need their own format. These are
    merged into a single .format() call -- calling .format() twice on a
    Styler resets the formatter for any column not in the second call, which
    silently undoes the whole-number formatting on value_cols.
    """
    def _white_for_nan(s):
        return ['background-color: #ffffff' if pd.isna(v) else '' for v in s]

    styler = df.style.background_gradient(subset=value_cols, cmap='RdYlGn', axis=None)
    for col in value_cols:
        styler = styler.apply(_white_for_nan, subset=[col])

    format_dict = {c: fmt for c in value_cols}
    if extra_formats:
        format_dict.update(extra_formats)
    styler = styler.format(format_dict, na_rep=na_rep)
    return styler


def cohort_table_to_retention_rate(cohort_customers_df, milestone_cols):
    """
    Convert a customer-count cohort table (output of build_monthly_cohort_table
    with value='customers') into a retention-rate table: each milestone column
    becomes that milestone's count divided by the cohort's New Customers (M0).
    """
    rate_df = cohort_customers_df.copy()
    for col in milestone_cols:
        if col == 'M0':
            rate_df[col] = 1.0
        else:
            rate_df[col] = rate_df[col] / rate_df['New Customers'].replace(0, np.nan)
    return rate_df


def get_calendar_block(anchor_date, horizon_months):
    """
    Find the calendar block (quarter, half-year, or full year) that contains
    `anchor_date`, sized according to `horizon_months`:
      3  -> calendar quarter (Q1 Jan-Mar, Q2 Apr-Jun, Q3 Jul-Sep, Q4 Oct-Dec)
      6  -> calendar half (H1 Jan-Jun, H2 Jul-Dec)
      12 -> calendar year (Jan-Dec)
    Any other horizon falls back to a block starting at anchor_date's month.

    Returns (block_start, block_end_exclusive) as Timestamps.
    """
    anchor_date = pd.Timestamp(anchor_date)
    year = anchor_date.year
    month = anchor_date.month

    if horizon_months == 3:
        block_start_month = ((month - 1) // 3) * 3 + 1
    elif horizon_months == 6:
        block_start_month = 1 if month <= 6 else 7
    elif horizon_months == 12:
        block_start_month = 1
    else:
        block_start_month = month

    block_start = pd.Timestamp(year, block_start_month, 1)
    block_end = block_start + pd.DateOffset(months=horizon_months)
    return block_start, block_end


def calculate_actual_comparison_revenue(transactions, forecast_start, horizon_months, comparison_type='Previous Period'):
    """
    Actual revenue for the comparison window aligned to the calendar block
    (quarter/half/year, matching horizon_months) containing forecast_start,
    split into new-customer and existing-customer components (same
    definition as elsewhere in the dashboard: "new" = first transaction
    falls within the window).

    comparison_type:
      'Previous Period' -> the calendar block immediately preceding the
                            anchor block (e.g. 3mo horizon anchored in Q1 2024
                            -> Q4 2023)
      'Previous Year'   -> the same calendar block, one year earlier
                            (e.g. 3mo horizon anchored in Q1 2024 -> Q1 2023)
    """
    anchor_start, anchor_end = get_calendar_block(forecast_start, horizon_months)

    if comparison_type == 'Previous Year':
        start = anchor_start - pd.DateOffset(years=1)
        end = anchor_end - pd.DateOffset(years=1)
    else:  # Previous Period
        start = anchor_start - pd.DateOffset(months=horizon_months)
        end = anchor_start

    window_trans = transactions[(transactions['transaction_date'] >= start) & (transactions['transaction_date'] < end)]
    total_revenue = float(window_trans['amount'].sum())

    # First transaction date per customer across full history, to classify "new" within this window.
    # Map customer_id -> first_txn_date with a vectorized .map(Series), then compare with
    # plain vectorized boolean ops (was: a Python lambda invoked once per transaction row).
    first_txn_date = transactions.groupby('customer_id')['transaction_date'].min()
    customer_first_txn = window_trans['customer_id'].map(first_txn_date)
    is_new_in_window = (customer_first_txn >= start) & (customer_first_txn < end)
    new_revenue = float(window_trans.loc[is_new_in_window, 'amount'].sum())
    existing_revenue = total_revenue - new_revenue

    return {
        'total': total_revenue, 'new': new_revenue, 'existing': existing_revenue,
        'start': start, 'end': end
    }


# ============================================================================
# LOAD DATA
# ============================================================================
transactions, rfm_ref = load_data()

all_regions = sorted(transactions['store_region'].dropna().unique())
all_categories = sorted(transactions['category'].dropna().unique())
all_eom = sorted(set(
    transactions['transaction_date'].apply(lambda d: end_of_month(d))
), reverse=True)
eom_options = [d.date() for d in all_eom]

# ============================================================================
# TABS
# ============================================================================
tab_fin, tab_forecast, tab_rfm, tab_segment = st.tabs(["Financial", "Forecast", "RFM Analysis", "Segment Analysis"])

# ============================================================================
# RFM TAB
# ============================================================================
with tab_rfm:
    st.markdown("## RFM Analysis")

    # Row 1 — Cutoff Date only
    rd1, rd2 = st.columns([1, 3])
    with rd1:
        cutoff_date = pd.Timestamp(st.date_input(
            "Cutoff Date",
            value=transactions['transaction_date'].max().date(),
            min_value=transactions['transaction_date'].min().date(),
            max_value=transactions['transaction_date'].max().date(),
            key='rfm_cutoff'
        ))

    # Calculate RFM before filters so segment list is available
    rfm_data = calculate_rfm_as_of_date(transactions, cutoff_date)

    if len(rfm_data) == 0:
        st.error("No data available for selected date.")
    else:
        rfm_data, segment_colors = assign_segments(rfm_data, rfm_ref)
        available_segments = sorted([s for s in rfm_data['Segment'].unique() if pd.notna(s)])

        # Row 2 — Region & Category
        rr1, rr2 = st.columns([1, 1])
        with rr1:
            rfm_regions = st.multiselect(
                "Filter by Region",
                options=all_regions,
                default=all_regions,
                key='rfm_regions'
            )
        with rr2:
            rfm_categories = st.multiselect(
                "Filter by Category",
                options=all_categories,
                default=all_categories,
                key='rfm_categories'
            )

        # Row 3 — Segment
        rs1, rs2 = st.columns([3, 1])
        with rs1:
            selected_segments = st.multiselect(
                "Filter by Segment",
                options=available_segments,
                default=available_segments,
                key='rfm_segments'
            )

        trans_filtered = transactions[
            (transactions['store_region'].isin(rfm_regions)) &
            (transactions['category'].isin(rfm_categories))
        ]
        filtered_rfm = rfm_data[
            rfm_data['Segment'].isin(selected_segments) &
            rfm_data['customer_id'].isin(trans_filtered['customer_id'].unique())
        ]

        st.markdown(f"**{cutoff_date.strftime('%d.%m.%Y')}** — {filtered_rfm['customer_id'].nunique():,} customers")
        st.markdown("---")

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_card("Customers", filtered_rfm['customer_id'].nunique(), fmt='integer')
        with k2:
            metric_card("Total Spend", filtered_rfm['Monetary'].sum(), fmt='currency')
        with k3:
            metric_card("Avg Frequency", filtered_rfm['Frequency'].mean(), fmt='default')
        with k4:
            metric_card("Avg Recency", filtered_rfm['Recency'].mean(skipna=True), fmt='default')

        st.markdown("---")

        # Segment charts
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Revenue by Segment")
            seg_revenue = filtered_rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
            colors_list = [segment_colors.get(s, {}).get('bg', '#3b82f6') for s in seg_revenue.index]
            st.plotly_chart(bar_chart(seg_revenue.index, seg_revenue.values, colors_list), use_container_width=True)

        with c2:
            st.markdown("### Customers by Segment")
            seg_count = filtered_rfm['Segment'].value_counts()
            colors_list = [segment_colors.get(s, {}).get('bg', '#3b82f6') for s in seg_count.index]
            st.plotly_chart(bar_chart(seg_count.index, seg_count.values, colors_list, fmt='count'), use_container_width=True)

        st.markdown("---")

        # Segment details
        st.markdown("### Segment Details")
        seg_stats = filtered_rfm.groupby('Segment').agg(
            Recency=('Recency', 'mean'), Frequency=('Frequency', 'mean'),
            Monetary=('Monetary', 'mean'), Count=('customer_id', 'count'),
            Avg_R=('R_Score', 'mean'), Avg_F=('F_Score', 'mean'), Avg_M=('M_Score', 'mean')
        ).round(1).sort_values('Count', ascending=False)
        seg_stats.columns = ['Recency', 'Frequency', 'Monetary', 'Count', 'Avg R', 'Avg F', 'Avg M']
        st.dataframe(seg_stats, use_container_width=True)

        st.markdown("---")

        # Customer table
        st.markdown("### Customer RFM Details")
        customer_table = filtered_rfm[[
            'customer_id', 'Last_Transaction_Date', 'Recency', 'Frequency', 'Monetary',
            'R_Score', 'F_Score', 'M_Score', 'Segment'
        ]].sort_values('Monetary', ascending=False).reset_index(drop=True)
        customer_table.columns = ['Customer ID', 'Last Txn', 'Recency', 'Frequency', 'Monetary €', 'R', 'F', 'M', 'Segment']
        st.dataframe(customer_table, use_container_width=True, height=500, hide_index=True)

        st.markdown("---")

        # Pattern rules
        st.markdown("### RFM Pattern Rules")
        if not rfm_ref.empty:
            display_cols = [c for c in ['Ranking', 'Category', 'RFM Pattern', 'Description'] if c in rfm_ref.columns]
            st.dataframe(rfm_ref[display_cols], use_container_width=True, hide_index=True)


# ============================================================================
# FINANCIAL TAB
# ============================================================================
def color_delta(val):
    """Shared styler for Δ% columns in comparison tables."""
    if val > 0:
        color = '#059669'
    elif val < 0:
        color = '#dc2626'
    else:
        color = '#6b7280'
    return f'color: {color}; font-weight: 700;'

with tab_fin:
    st.markdown("## Financial Dashboard")

    # Controls row
    fc1, fc2, fc3 = st.columns([2, 2, 2])
    with fc1:
        selected_eom = st.selectbox(
            "Reference End of Month",
            options=eom_options,
            format_func=lambda d: pd.Timestamp(d).strftime('%b %Y'),
            key='fin_eom'
        )
        selected_eom = pd.Timestamp(selected_eom)

    with fc2:
        period_name = st.selectbox(
            "Period",
            options=['CM', 'This QTR', 'Last QTR', 'YTD', 'LTM', 'LY'],
            key='fin_period'
        )

    with fc3:
        comparison_type = st.radio(
            "Compare to",
            options=['Previous Period', 'Previous Year'],
            horizontal=True,
            key='fin_compare'
        )

    curr_start, curr_end = get_period_dates(period_name, selected_eom)
    comp_start, comp_end = get_comparison_dates(period_name, curr_start, curr_end, comparison_type)

    st.caption(
        f"**Current:** {curr_start.strftime('%d.%m.%Y')} → {curr_end.strftime('%d.%m.%Y')}"
        f"  |  **Comparison:** {comp_start.strftime('%d.%m.%Y')} → {comp_end.strftime('%d.%m.%Y')}"
    )

    st.markdown("---")

    # Filters row 1 — Region & Category
    ff1, ff2 = st.columns([1, 1])
    with ff1:
        fin_regions = st.multiselect(
            "Filter by Region",
            options=all_regions,
            default=all_regions,
            key='fin_regions'
        )
    with ff2:
        fin_categories = st.multiselect(
            "Filter by Category",
            options=all_categories,
            default=all_categories,
            key='fin_categories'
        )

    # RFM + segments as of selected_eom — computed once, reused below for both
    # the segment-filter options and the segment charts (was computed twice)
    rfm_fin = calculate_rfm_as_of_date(transactions, selected_eom)
    if len(rfm_fin) > 0:
        rfm_fin, seg_colors_fin = assign_segments(rfm_fin, rfm_ref)
        available_fin_segments = sorted([s for s in rfm_fin['Segment'].unique() if pd.notna(s)])
    else:
        seg_colors_fin = {}
        available_fin_segments = []

    # Filters row 2 — Segment
    fs1, fs2 = st.columns([3, 1])
    with fs1:
        fin_segments = st.multiselect(
            "Filter by Segment",
            options=available_fin_segments,
            default=available_fin_segments,
            key='fin_segments'
        )

    def filter_trans(start, end, include_segments=True):
        trans = transactions[
            (transactions['transaction_date'] >= start) &
            (transactions['transaction_date'] <= end) &
            (transactions['store_region'].isin(fin_regions)) &
            (transactions['category'].isin(fin_categories))
        ]
        
        # Apply segment filter if requested
        if include_segments and fin_segments:
            # Get RFM data and segment assignments as of period end
            rfm_for_seg = calculate_rfm_as_of_date(transactions, end)
            if len(rfm_for_seg) > 0:
                rfm_for_seg, _ = assign_segments(rfm_for_seg, rfm_ref)
                seg_customers = rfm_for_seg[rfm_for_seg['Segment'].isin(fin_segments)]['customer_id'].unique()
                trans = trans[trans['customer_id'].isin(seg_customers)]
        
        return trans

    trans_curr = filter_trans(curr_start, curr_end)
    trans_comp = filter_trans(comp_start, comp_end)

    # Segment-enriched views — computed once, reused across all charts
    trans_curr_seg = trans_curr.merge(rfm_fin[['customer_id', 'Segment']], on='customer_id', how='left')
    trans_curr_seg['Segment'] = trans_curr_seg['Segment'].fillna('Unknown')
    trans_comp_seg = trans_comp.merge(rfm_fin[['customer_id', 'Segment']], on='customer_id', how='left')
    trans_comp_seg['Segment'] = trans_comp_seg['Segment'].fillna('Unknown')

    # KPI cards
    st.markdown("### Key Metrics")
    
    # Calculate pre-periods for churn calculation
    pre_curr_start, pre_curr_end = get_comparison_dates(period_name, curr_start, curr_end, comparison_type)
    pre_comp_start, pre_comp_end = get_comparison_dates(period_name, comp_start, comp_end, comparison_type)
    
    # Calculate customer cohorts (respects region, category, segment filters, and period logic)
    cohort_metrics = calculate_customer_cohorts(
        transactions, 
        curr_start, curr_end, comp_start, comp_end,
        pre_curr_start, pre_curr_end, pre_comp_start, pre_comp_end,
        regions=fin_regions, categories=fin_categories
    )
    
    # First row: Core metrics
    kc1, kc2, kc3 = st.columns(3)
    with kc1:
        metric_card("Revenue", trans_curr['amount'].sum(), trans_comp['amount'].sum(), fmt='currency')
    with kc2:
        metric_card("Transactions", len(trans_curr), len(trans_comp), fmt='integer')
    with kc3:
        metric_card("Active Customers", trans_curr['customer_id'].nunique(), trans_comp['customer_id'].nunique(), fmt='integer')
    
    # Second row: Customer cohort metrics
    kc4, kc5, kc6, kc7 = st.columns(4)
    with kc4:
        metric_card("New Customers", cohort_metrics['new_curr'], cohort_metrics['new_comp'], fmt='integer')
    with kc5:
        metric_card("Acquisition Rate", cohort_metrics['acq_rate_curr'], cohort_metrics['acq_rate_comp'], fmt='percentage')
    with kc6:
        metric_card("Churned Customers", cohort_metrics['churned_curr'], cohort_metrics['churned_comp'], fmt='integer')
    with kc7:
        metric_card("Churn Rate", cohort_metrics['churn_rate_curr'], cohort_metrics['churn_rate_comp'], fmt='percentage')

    st.markdown("---")

    # RFM Segment charts (rfm_fin / seg_colors_fin already computed above)
    st.markdown("### Revenue & Customers by RFM Segment")
    if len(rfm_fin) > 0:
        sc1, sc2 = st.columns(2)
        with sc1:
            seg_rev = trans_curr_seg.groupby('Segment')['amount'].sum().sort_values(ascending=False)
            colors_list = [seg_colors_fin.get(s, {}).get('bg', '#3b82f6') for s in seg_rev.index]
            st.plotly_chart(bar_chart(seg_rev.index, seg_rev.values, colors_list, title='Revenue by Segment'), use_container_width=True)

        with sc2:
            seg_cust = trans_curr_seg.groupby('Segment')['customer_id'].nunique().sort_values(ascending=False)
            colors_list = [seg_colors_fin.get(s, {}).get('bg', '#3b82f6') for s in seg_cust.index]
            st.plotly_chart(bar_chart(seg_cust.index, seg_cust.values, colors_list, title='Active Customers by Segment', fmt='count'), use_container_width=True)

    st.markdown("---")

    # Revenue by Category
    st.markdown("### Revenue by Category")
    cat_view = st.radio("View", options=['Side-by-Side', 'Stacked by Segment'], horizontal=True, key='cat_view')
    
    period_label = f"Current: {curr_start.strftime('%d.%m.%Y')} – {curr_end.strftime('%d.%m.%Y')} | Comparison: {comp_start.strftime('%d.%m.%Y')} – {comp_end.strftime('%d.%m.%Y')}"
    st.caption(period_label)
    
    if cat_view == 'Side-by-Side':
        cat_rev_curr = trans_curr.groupby('category')['amount'].sum().sort_values(ascending=False)
        cat_rev_comp = trans_comp.groupby('category')['amount'].sum()
        
        # Align comparison to current order
        cat_rev_comp = cat_rev_comp.reindex(cat_rev_curr.index, fill_value=0)
        
        total_curr = cat_rev_curr.sum()
        total_comp = cat_rev_comp.sum()
        
        fig_cat = go.Figure()
        
        # Current bars
        pct_curr = (cat_rev_curr / total_curr * 100) if total_curr > 0 else 0
        hover_curr = [f"{cat}<br>Current: €{val:,.0f} ({pct:.1f}%)" for cat, val, pct in zip(cat_rev_curr.index, cat_rev_curr.values, pct_curr)]
        
        fig_cat.add_trace(go.Bar(
            name="Current",
            x=cat_rev_curr.index,
            y=cat_rev_curr.values,
            marker=dict(color='#0066cc', opacity=1.0),
            offsetgroup=0,
            hovertemplate='%{customdata[0]}<extra></extra>',
            customdata=hover_curr,
            text=[f"€{v:,.0f}<br>{p:.1f}%" for v, p in zip(cat_rev_curr.values, pct_curr)],
            textposition='inside',
            textfont=dict(size=10, color='white'),
            showlegend=False
        ))
        
        # Comparison bars
        pct_comp = (cat_rev_comp / total_comp * 100) if total_comp > 0 else 0
        hover_comp = [f"{cat}<br>Comparison: €{val:,.0f} ({pct:.1f}%)" for cat, val, pct in zip(cat_rev_comp.index, cat_rev_comp.values, pct_comp)]
        
        fig_cat.add_trace(go.Bar(
            name="Comparison",
            x=cat_rev_comp.index,
            y=cat_rev_comp.values,
            marker=dict(color='#0066cc', opacity=0.4),
            offsetgroup=1,
            hovertemplate='%{customdata[0]}<extra></extra>',
            customdata=hover_comp,
            text=[f"€{v:,.0f}<br>{p:.1f}%" for v, p in zip(cat_rev_comp.values, pct_comp)],
            textposition='inside',
            textfont=dict(size=10, color='white'),
            showlegend=False
        ))
        
        fig_cat.update_layout(
            barmode='group',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=60, b=20, l=20, r=20),
            yaxis_gridcolor='#e5e7eb',
            yaxis_title='Revenue (€)',
            showlegend=False
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        # Stacked by Segment - both Current and Comparison
        cat_rev_curr_seg = trans_curr_seg.groupby(['category', 'Segment'])['amount'].sum().reset_index()
        cat_rev_comp_seg = trans_comp_seg.groupby(['category', 'Segment'])['amount'].sum().reset_index()
        
        # Get all categories in order (by total revenue)
        all_cats = sorted(set(cat_rev_curr_seg['category']) | set(cat_rev_comp_seg['category']))
        cat_totals = {}
        for cat in all_cats:
            cat_totals[cat] = (cat_rev_curr_seg[cat_rev_curr_seg['category'] == cat]['amount'].sum() + 
                               cat_rev_comp_seg[cat_rev_comp_seg['category'] == cat]['amount'].sum())
        all_cats_sorted = sorted(all_cats, key=lambda x: cat_totals.get(x, 0), reverse=True)
        
        all_segments = sorted(set(cat_rev_curr_seg['Segment']) | set(cat_rev_comp_seg['Segment']))
        
        # Create x-axis labels
        x_labels = []
        for cat in all_cats_sorted:
            x_labels.extend([f"{cat}<br>(Curr)", f"{cat}<br>(Comp)"])
        
        fig_cat_stack = go.Figure()
        
        # For each segment, create traces with both Current and Comparison
        for segment in all_segments:
            # Get data for this segment across all categories
            curr_data = {}
            comp_data = {}
            
            for cat in all_cats_sorted:
                curr_val = cat_rev_curr_seg[
                    (cat_rev_curr_seg['category'] == cat) & 
                    (cat_rev_curr_seg['Segment'] == segment)
                ]['amount'].sum()
                comp_val = cat_rev_comp_seg[
                    (cat_rev_comp_seg['category'] == cat) & 
                    (cat_rev_comp_seg['Segment'] == segment)
                ]['amount'].sum()
                
                curr_data[cat] = curr_val
                comp_data[cat] = comp_val
            
            # Build y values (alternating curr, comp, curr, comp...)
            y_vals = []
            for cat in all_cats_sorted:
                y_vals.append(curr_data[cat])
                y_vals.append(comp_data[cat])
            
            # Build opacities (alternating 1.0, 0.4, 1.0, 0.4...)
            opacities = [1.0 if i % 2 == 0 else 0.4 for i in range(len(y_vals))]
            
            # Calculate percentages (% of total bar per Curr/Comp column)
            cat_totals_curr = {}
            cat_totals_comp = {}
            for cat in all_cats_sorted:
                cat_totals_curr[cat] = cat_rev_curr_seg[cat_rev_curr_seg['category'] == cat]['amount'].sum()
                cat_totals_comp[cat] = cat_rev_comp_seg[cat_rev_comp_seg['category'] == cat]['amount'].sum()
            
            pcts = []
            for i, val in enumerate(y_vals):
                cat_idx = i // 2
                cat = all_cats_sorted[cat_idx]
                if i % 2 == 0:
                    pct = (val / cat_totals_curr[cat] * 100) if cat_totals_curr[cat] > 0 else 0
                else:
                    pct = (val / cat_totals_comp[cat] * 100) if cat_totals_comp[cat] > 0 else 0
                pcts.append(pct)
            
            color = seg_colors_fin.get(segment, {}).get('bg', '#3b82f6')
            
            fig_cat_stack.add_trace(go.Bar(
                name=segment,
                x=x_labels,
                y=y_vals,
                marker=dict(
                    color=[color] * len(y_vals),
                    opacity=opacities
                ),
                text=[f"€{v:,.0f}<br>{p:.1f}%" if v > 0 else "" for v, p in zip(y_vals, pcts)],
                textposition='inside',
                textfont=dict(size=9, color='white'),
                hoverinfo='skip',
                showlegend=False
            ))
        
        fig_cat_stack.update_layout(
            barmode='stack',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=40, b=20, l=20, r=20),
            yaxis_gridcolor='#e5e7eb',
            yaxis_title='Revenue (€)',
            showlegend=False,
            hovermode='x unified'
        )
        st.plotly_chart(fig_cat_stack, use_container_width=True)

    st.markdown("---")

    # Revenue by Region
    st.markdown("### Revenue by Region")
    reg_view = st.radio("View", options=['Side-by-Side', 'Stacked by Segment'], horizontal=True, key='reg_view')
    
    st.caption(period_label)
    
    if reg_view == 'Side-by-Side':
        reg_rev_curr = trans_curr.groupby('store_region')['amount'].sum().sort_values(ascending=False)
        reg_rev_comp = trans_comp.groupby('store_region')['amount'].sum()
        
        # Align comparison to current order
        reg_rev_comp = reg_rev_comp.reindex(reg_rev_curr.index, fill_value=0)
        
        total_curr = reg_rev_curr.sum()
        total_comp = reg_rev_comp.sum()
        
        fig_reg = go.Figure()
        
        # Current bars
        pct_curr = (reg_rev_curr / total_curr * 100) if total_curr > 0 else 0
        hover_curr = [f"{reg}<br>Current: €{val:,.0f} ({pct:.1f}%)" for reg, val, pct in zip(reg_rev_curr.index, reg_rev_curr.values, pct_curr)]
        
        fig_reg.add_trace(go.Bar(
            name="Current",
            x=reg_rev_curr.index,
            y=reg_rev_curr.values,
            marker=dict(color='#059669', opacity=1.0),
            offsetgroup=0,
            hovertemplate='%{customdata[0]}<extra></extra>',
            customdata=hover_curr,
            text=[f"€{v:,.0f}<br>{p:.1f}%" for v, p in zip(reg_rev_curr.values, pct_curr)],
            textposition='inside',
            textfont=dict(size=10, color='white'),
            showlegend=False
        ))
        
        # Comparison bars
        pct_comp = (reg_rev_comp / total_comp * 100) if total_comp > 0 else 0
        hover_comp = [f"{reg}<br>Comparison: €{val:,.0f} ({pct:.1f}%)" for reg, val, pct in zip(reg_rev_comp.index, reg_rev_comp.values, pct_comp)]
        
        fig_reg.add_trace(go.Bar(
            name="Comparison",
            x=reg_rev_comp.index,
            y=reg_rev_comp.values,
            marker=dict(color='#059669', opacity=0.4),
            offsetgroup=1,
            hovertemplate='%{customdata[0]}<extra></extra>',
            customdata=hover_comp,
            text=[f"€{v:,.0f}<br>{p:.1f}%" for v, p in zip(reg_rev_comp.values, pct_comp)],
            textposition='inside',
            textfont=dict(size=10, color='white'),
            showlegend=False
        ))
        
        fig_reg.update_layout(
            barmode='group',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=60, b=20, l=20, r=20),
            yaxis_gridcolor='#e5e7eb',
            yaxis_title='Revenue (€)',
            showlegend=False
        )
        st.plotly_chart(fig_reg, use_container_width=True)
    else:
        # Stacked by Segment - both Current and Comparison
        reg_rev_curr_seg = trans_curr_seg.groupby(['store_region', 'Segment'])['amount'].sum().reset_index()
        reg_rev_comp_seg = trans_comp_seg.groupby(['store_region', 'Segment'])['amount'].sum().reset_index()
        
        # Get all regions in order (by total revenue)
        all_regs = sorted(set(reg_rev_curr_seg['store_region']) | set(reg_rev_comp_seg['store_region']))
        reg_totals = {}
        for reg in all_regs:
            reg_totals[reg] = (reg_rev_curr_seg[reg_rev_curr_seg['store_region'] == reg]['amount'].sum() + 
                               reg_rev_comp_seg[reg_rev_comp_seg['store_region'] == reg]['amount'].sum())
        all_regs_sorted = sorted(all_regs, key=lambda x: reg_totals.get(x, 0), reverse=True)
        
        all_segments = sorted(set(reg_rev_curr_seg['Segment']) | set(reg_rev_comp_seg['Segment']))
        
        # Create x-axis labels
        x_labels = []
        for reg in all_regs_sorted:
            x_labels.extend([f"{reg}<br>(Curr)", f"{reg}<br>(Comp)"])
        
        fig_reg_stack = go.Figure()
        
        # For each segment, create traces with both Current and Comparison
        for segment in all_segments:
            # Get data for this segment across all regions
            curr_data = {}
            comp_data = {}
            
            for reg in all_regs_sorted:
                curr_val = reg_rev_curr_seg[
                    (reg_rev_curr_seg['store_region'] == reg) & 
                    (reg_rev_curr_seg['Segment'] == segment)
                ]['amount'].sum()
                comp_val = reg_rev_comp_seg[
                    (reg_rev_comp_seg['store_region'] == reg) & 
                    (reg_rev_comp_seg['Segment'] == segment)
                ]['amount'].sum()
                
                curr_data[reg] = curr_val
                comp_data[reg] = comp_val
            
            # Build y values (alternating curr, comp, curr, comp...)
            y_vals = []
            for reg in all_regs_sorted:
                y_vals.append(curr_data[reg])
                y_vals.append(comp_data[reg])
            
            # Build opacities (alternating 1.0, 0.4, 1.0, 0.4...)
            opacities = [1.0 if i % 2 == 0 else 0.4 for i in range(len(y_vals))]
            
            # Calculate percentages (% of total bar per Curr/Comp column)
            reg_totals_curr = {}
            reg_totals_comp = {}
            for reg in all_regs_sorted:
                reg_totals_curr[reg] = reg_rev_curr_seg[reg_rev_curr_seg['store_region'] == reg]['amount'].sum()
                reg_totals_comp[reg] = reg_rev_comp_seg[reg_rev_comp_seg['store_region'] == reg]['amount'].sum()
            
            pcts = []
            for i, val in enumerate(y_vals):
                reg_idx = i // 2
                reg = all_regs_sorted[reg_idx]
                if i % 2 == 0:
                    pct = (val / reg_totals_curr[reg] * 100) if reg_totals_curr[reg] > 0 else 0
                else:
                    pct = (val / reg_totals_comp[reg] * 100) if reg_totals_comp[reg] > 0 else 0
                pcts.append(pct)
            
            color = seg_colors_fin.get(segment, {}).get('bg', '#3b82f6')
            
            fig_reg_stack.add_trace(go.Bar(
                name=segment,
                x=x_labels,
                y=y_vals,
                marker=dict(
                    color=[color] * len(y_vals),
                    opacity=opacities
                ),
                text=[f"€{v:,.0f}<br>{p:.1f}%" if v > 0 else "" for v, p in zip(y_vals, pcts)],
                textposition='inside',
                textfont=dict(size=9, color='white'),
                hoverinfo='skip',
                showlegend=False
            ))
        
        fig_reg_stack.update_layout(
            barmode='stack',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=40, b=20, l=20, r=20),
            yaxis_gridcolor='#e5e7eb',
            yaxis_title='Revenue (€)',
            showlegend=False,
            hovermode='x unified'
        )
        st.plotly_chart(fig_reg_stack, use_container_width=True)

    st.markdown("---")

    # Category comparison table
    st.markdown("### Category Detail — Period Comparison")
    cat_rows = []
    for cat in sorted(fin_categories):
        curr_v = trans_curr[trans_curr['category'] == cat]['amount'].sum()
        comp_v = trans_comp[trans_comp['category'] == cat]['amount'].sum()
        chg = ((curr_v - comp_v) / comp_v * 100) if comp_v > 0 else 0
        cat_rows.append({'Category': cat, 'Current': curr_v, 'Comparison': comp_v, 'Δ%': chg})
    cat_df = pd.DataFrame(cat_rows)
    
    styled_cat_df = cat_df.style.format({'Current': '€{:,.2f}', 'Comparison': '€{:,.2f}', 'Δ%': '{:+.1f}%'}).map(
        color_delta,
        subset=['Δ%']
    )
    st.dataframe(styled_cat_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Region comparison table
    st.markdown("### Region Detail — Period Comparison")
    reg_rows = []
    for reg in sorted(fin_regions):
        curr_v = trans_curr[trans_curr['store_region'] == reg]['amount'].sum()
        comp_v = trans_comp[trans_comp['store_region'] == reg]['amount'].sum()
        chg = ((curr_v - comp_v) / comp_v * 100) if comp_v > 0 else 0
        reg_rows.append({'Region': reg, 'Current': curr_v, 'Comparison': comp_v, 'Δ%': chg})
    reg_df = pd.DataFrame(reg_rows)
    
    styled_reg_df = reg_df.style.format({'Current': '€{:,.2f}', 'Comparison': '€{:,.2f}', 'Δ%': '{:+.1f}%'}).map(
        color_delta,
        subset=['Δ%']
    )
    st.dataframe(styled_reg_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ============================================================================
    # MONTHLY TRENDS SECTION
    # ============================================================================
    st.markdown("## Monthly Trends")
    
    # Period controls: range selector
    all_months = sorted(transactions['transaction_date'].dt.to_period('M').unique())
    month_labels = [str(m) for m in all_months]  # Format: '2024-01'
    month_display = [pd.Timestamp(m).strftime('%b %Y') for m in month_labels]  # Format: 'Jan 2024'
    
    # Range selector for month selection (two-point slider)
    selected_range = st.select_slider(
        "Select Period",
        options=range(len(month_display)),
        value=(max(0, len(month_labels) - 24), len(month_labels) - 1),
        format_func=lambda i: month_display[i],
        key='mt_month_range'
    )
    
    start_idx, end_idx = selected_range
    start_month = pd.Timestamp(month_labels[start_idx])
    end_month = pd.Timestamp(month_labels[end_idx])
    # Adjust to end of month
    end_month = (end_month + pd.DateOffset(months=1)).replace(day=1) - pd.Timedelta(days=1)
    
    mt_comparison = st.radio(
        "Compare to",
        options=['None', 'Previous Year'],
        horizontal=True,
        key='mt_compare'
    )
    
    # Validate date range
    if start_month > end_month:
        st.error("Start Month must be before End Month")
    else:
        st.caption(f"**Period:** {start_month.strftime('%b %Y')} → {end_month.strftime('%b %Y')}")
        st.markdown("---")
        
        # Calculate metrics for selected period
        monthly_data = calculate_monthly_metrics(
            transactions,
            start_month,
            end_month,
            regions=fin_regions,
            categories=fin_categories
        )
        
        # Reformat month labels for display (avoid date interpretation by Plotly)
        if len(monthly_data) > 0:
            monthly_data['Month_Display'] = monthly_data['Month'].apply(lambda x: pd.Timestamp(x).strftime('%b %y'))
        else:
            st.info("No transactions found for the selected period and filters.")
            st.markdown("---")
            st.stop()
        
        # Calculate comparison period
        comparison_data = None
        if mt_comparison == 'Previous Year':
            comp_start = start_month - pd.DateOffset(years=1)
            comp_end = end_month - pd.DateOffset(years=1)
            comparison_data = calculate_monthly_metrics(
                transactions,
                comp_start,
                comp_end,
                regions=fin_regions,
                categories=fin_categories
            )
            if len(comparison_data) > 0:
                # For Previous Year comparison, use just month name (Jan, Feb) to align properly
                comparison_data['Month_Display'] = comparison_data['Month'].apply(lambda x: pd.Timestamp(x).strftime('%b'))
        
        # Format month display - for Previous Year, use just month names (Jan, Feb) to align
        if mt_comparison == 'Previous Year' and comparison_data is not None and len(comparison_data) > 0:
            monthly_data['Month_Display'] = monthly_data['Month'].apply(lambda x: pd.Timestamp(x).strftime('%b'))
            comparison_label = 'Previous Year'
        else:
            monthly_data['Month_Display'] = monthly_data['Month'].apply(lambda x: pd.Timestamp(x).strftime('%b %y'))
            comparison_label = 'Comparison'
        
        # ---- Chart 1: Monthly Revenue (Bar) ----
        st.markdown("### Monthly Revenue")
        fig_rev = go.Figure()
        
        fig_rev.add_trace(go.Bar(
            x=monthly_data['Month_Display'],
            y=monthly_data['Revenue'],
            name='Current',
            marker=dict(color='#0066cc'),
            hovertemplate='%{x}<br>€%{y:,.0f}<extra></extra>'
        ))
        
        if comparison_data is not None and len(comparison_data) > 0:
            fig_rev.add_trace(go.Bar(
                x=comparison_data['Month_Display'],
                y=comparison_data['Revenue'],
                name=comparison_label,
                marker=dict(color='#b3d9ff'),
                hovertemplate='%{x}<br>€%{y:,.0f}<extra></extra>'
            ))
        
        fig_rev.update_layout(
            height=320,
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=40, b=20, l=20, r=20),
            yaxis_gridcolor='#e5e7eb',
            yaxis_title='Revenue (€)',
            xaxis_title='',
            hovermode='x unified',
            barmode='group'
        )
        st.plotly_chart(fig_rev, use_container_width=True)
        
        # ---- Chart 2: Monthly Transactions (Bar) ----
        st.markdown("### Monthly Transactions")
        fig_txn = go.Figure()
        
        fig_txn.add_trace(go.Bar(
            x=monthly_data['Month_Display'],
            y=monthly_data['Transactions'],
            name='Current',
            marker=dict(color='#0066cc'),
            hovertemplate='%{x}<br>%{y:,.0f}<extra></extra>'
        ))
        
        if comparison_data is not None and len(comparison_data) > 0:
            fig_txn.add_trace(go.Bar(
                x=comparison_data['Month_Display'],
                y=comparison_data['Transactions'],
                name=comparison_label,
                marker=dict(color='#b3d9ff'),
                hovertemplate='%{x}<br>%{y:,.0f}<extra></extra>'
            ))
        
        fig_txn.update_layout(
            height=320,
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=40, b=20, l=20, r=20),
            yaxis_gridcolor='#e5e7eb',
            yaxis_title='Transaction Count',
            xaxis_title='',
            hovermode='x unified',
            barmode='group'
        )
        st.plotly_chart(fig_txn, use_container_width=True)
        
        # ---- Chart 3: Monthly Active Customers (Bar) ----
        st.markdown("### Monthly Active Customers")
        fig_cust = go.Figure()
        
        fig_cust.add_trace(go.Bar(
            x=monthly_data['Month_Display'],
            y=monthly_data['Active_Customers'],
            name='Current',
            marker=dict(color='#059669'),
            hovertemplate='%{x}<br>%{y:,.0f}<extra></extra>'
        ))
        
        if comparison_data is not None and len(comparison_data) > 0:
            fig_cust.add_trace(go.Bar(
                x=comparison_data['Month_Display'],
                y=comparison_data['Active_Customers'],
                name=comparison_label,
                marker=dict(color='#a7f3d0'),
                hovertemplate='%{x}<br>%{y:,.0f}<extra></extra>'
            ))
        
        fig_cust.update_layout(
            height=320,
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=40, b=20, l=20, r=20),
            yaxis_gridcolor='#e5e7eb',
            yaxis_title='Active Customers',
            xaxis_title='',
            hovermode='x unified',
            barmode='group'
        )
        st.plotly_chart(fig_cust, use_container_width=True)
        
        # Summary Statistics (using metric_card style)
        st.markdown("### Summary Statistics")
        summary_cols = st.columns(3)
        
        show_delta = comparison_data is not None and len(comparison_data) > 0
        
        with summary_cols[0]:
            metric_card(
                "Total Revenue",
                monthly_data['Revenue'].sum(),
                comparison_data['Revenue'].sum() if show_delta else None,
                fmt='currency'
            )
        
        with summary_cols[1]:
            metric_card(
                "Total Transactions",
                monthly_data['Transactions'].sum(),
                comparison_data['Transactions'].sum() if show_delta else None,
                fmt='integer'
            )
        
        with summary_cols[2]:
            metric_card(
                "Avg Monthly Customers",
                monthly_data['Active_Customers'].mean(),
                comparison_data['Active_Customers'].mean() if show_delta else None,
                fmt='integer'
            )


# ============================================================================
# SEGMENT ANALYSIS TAB
# ============================================================================
with tab_segment:
    st.markdown("## Segment Analysis")

    # Controls row 1 — Date & Analysis Window
    sc1, sc2, sc3 = st.columns([2, 2, 2])
    with sc1:
        selected_eom_seg = st.selectbox(
            "Reference End of Month",
            options=eom_options,
            format_func=lambda d: pd.Timestamp(d).strftime('%b %Y'),
            key='seg_eom'
        )
        selected_eom_seg = pd.Timestamp(selected_eom_seg)

    with sc2:
        lookback_months_seg = st.selectbox(
            "Analysis Window",
            options=[3, 6, 12],
            index=2,
            key='seg_lookback'
        )

    with sc3:
        st.write("")  # Spacer
    
    # Controls row 2 — Region Filter (half width)
    sr1, sr2 = st.columns([1, 1])
    with sr1:
        seg_regions = st.multiselect(
            "Filter by Region",
            options=all_regions,
            default=all_regions,
            key='seg_regions'
        )

    with sr2:
        st.write("")  # Spacer

    analysis_start = (selected_eom_seg - pd.DateOffset(months=lookback_months_seg - 1)).replace(day=1)
    
    st.markdown(f"**Analysis Period:** {analysis_start.strftime('%b %Y')} → {selected_eom_seg.strftime('%b %Y')} ({lookback_months_seg} months)")
    st.markdown("---")

    # ---- Section 1: Segment Health Scorecard ----
    st.markdown("### Segment Health Scorecard")
    st.caption(f"Growth % = change in customer count over {lookback_months_seg}-month window")
    
    scorecard = get_segment_health_scorecard(transactions, rfm_ref, selected_eom_seg, lookback_months_seg, regions=seg_regions)
    
    if len(scorecard) > 0:
        # Format for display
        scorecard_display = scorecard.copy()
        scorecard_display['Growth %'] = scorecard_display['Growth %'].apply(lambda x: f"{x:+.1f}%")
        scorecard_display['Avg Revenue €'] = scorecard_display['Avg Revenue €'].apply(lambda x: f"€{x:,.0f}")
        scorecard_display['Revenue %'] = scorecard_display['Revenue %'].apply(lambda x: f"{x:.1f}%")
        scorecard_display['Avg Recency Days'] = scorecard_display['Avg Recency Days'].apply(lambda x: f"{x:.0f}")
        
        # Apply color styling to Growth % column
        def color_growth(val_str):
            try:
                val = float(val_str.replace('%', '').replace('+', ''))
                if val > 0:
                    return 'color: #059669; font-weight: 700;'
                elif val < 0:
                    return 'color: #dc2626; font-weight: 700;'
            except:
                pass
            return ''
        
        styled_df = scorecard_display.style.map(
            lambda x: color_growth(str(x)) if isinstance(x, str) and '%' in str(x) else '',
            subset=['Growth %']
        )
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.caption("Avg Recency = days since last transaction (lower = healthier)")
    else:
        st.info("No segment data available for selected date.")

    st.markdown("---")

    # ---- Section 2: Segment Trends (Revenue & Customer Count) ----
    st.markdown(f"### Segment Trends (Last {lookback_months_seg} Months)")
    
    months_axis, revenue_trends, count_trends = get_segment_monthly_trends(transactions, rfm_ref, selected_eom_seg, lookback_months_seg, regions=seg_regions)
    
    if months_axis:
        # Revenue stacked area chart
        fig_rev_trend = go.Figure()
        
        # Get segment colors
        rfm_temp = calculate_rfm_as_of_date(transactions, selected_eom_seg)
        if len(rfm_temp) > 0:
            rfm_temp, seg_colors_temp = assign_segments(rfm_temp, rfm_ref)
        else:
            seg_colors_temp = {}
        
        for segment in sorted(revenue_trends.keys()):
            if segment in ['Unknown', 'New Customer']:
                continue
            color = seg_colors_temp.get(segment, {}).get('bg', '#3b82f6')
            fig_rev_trend.add_trace(go.Scatter(
                x=months_axis,
                y=revenue_trends[segment],
                name=segment,
                mode='lines',
                line=dict(color=color, width=2),
                stackgroup='one',
                hovertemplate=f"{segment}<br>%{{x}}<br>€%{{y:,.0f}}<extra></extra>"
            ))
        
        fig_rev_trend.update_layout(
            height=350, plot_bgcolor='rgba(0,0,0,0)', 
            margin=dict(t=40, b=20, l=20, r=20),
            yaxis_gridcolor='#e5e7eb', xaxis_gridcolor='#f3f4f6',
            hovermode='x unified',
            yaxis_title='Monthly Revenue (€)',
            xaxis_title='Month'
        )
        st.plotly_chart(fig_rev_trend, use_container_width=True)

        # Customer count trend
        st.markdown(f"### Customer Count Trend by Segment")
        fig_count_trend = go.Figure()
        
        for segment in sorted(count_trends.keys()):
            if segment in ['Unknown', 'New Customer']:
                continue
            color = seg_colors_temp.get(segment, {}).get('bg', '#3b82f6')
            fig_count_trend.add_trace(go.Scatter(
                x=months_axis,
                y=count_trends[segment],
                name=segment,
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f"{segment}<br>%{{x}}<br>%{{y:,.0f}} customers<extra></extra>"
            ))
        
        fig_count_trend.update_layout(
            height=350, plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=40, b=20, l=20, r=20),
            yaxis_gridcolor='#e5e7eb', xaxis_gridcolor='#f3f4f6',
            hovermode='x unified',
            yaxis_title='Active Customers',
            xaxis_title='Month'
        )
        st.plotly_chart(fig_count_trend, use_container_width=True)

    st.markdown("---")

    # ---- Section 3: Segment Migration Matrix ----
    st.markdown(f"### Customer Migration Between Segments ({lookback_months_seg}-Month Window)")

    migration_matrix, counts_start, counts_end, counts_new = get_segment_migration_matrix(transactions, rfm_ref, analysis_start, selected_eom_seg, regions=seg_regions)

    if len(migration_matrix) > 0:
        start_label = analysis_start.strftime('%b %Y')
        end_label   = selected_eom_seg.strftime('%b %Y')

        all_segs = sorted(set(counts_start.index) | set(counts_end.index) | set(counts_new.index))
        # End-of-period existing (non-new) customers per segment, for the stack
        counts_end_existing = {seg: int(counts_end.get(seg, 0)) - int(counts_new.get(seg, 0)) for seg in all_segs}

        total_start = int(sum(int(counts_start.get(s, 0)) for s in all_segs))
        total_end   = int(sum(int(counts_end.get(s, 0)) for s in all_segs))
        total_new   = int(sum(int(counts_new.get(s, 0)) for s in all_segs))

        # -- Chart 1: Segment size, Start vs End (existing + new stacked) --
        st.markdown(f"**Segment size:** {start_label} vs {end_label}")

        fig_size = go.Figure()

        start_vals = [int(counts_start.get(s, 0)) for s in all_segs]
        end_existing_vals = [counts_end_existing.get(s, 0) for s in all_segs]
        end_new_vals = [int(counts_new.get(s, 0)) for s in all_segs]
        end_total_vals = [int(counts_end.get(s, 0)) for s in all_segs]

        bar_colors = [seg_colors_temp.get(s, {}).get('bg', '#3b82f6') for s in all_segs]

        # Start bar (lighter/translucent version of segment color)
        fig_size.add_trace(go.Bar(
            name=start_label,
            x=all_segs,
            y=start_vals,
            marker=dict(color=bar_colors, opacity=0.4),
            offsetgroup=0,
            showlegend=False,
            customdata=[[v, (v / total_start * 100) if total_start else 0] for v in start_vals],
            hovertemplate=f'%{{x}} · {start_label}<br>%{{customdata[0]:,}} customers (%{{customdata[1]:.1f}}% of total)<extra></extra>'
        ))

        # End bar, stacked: existing (solid) + new (hatched/distinct)
        fig_size.add_trace(go.Bar(
            name=f'{end_label} (existing)',
            x=all_segs,
            y=end_existing_vals,
            marker=dict(color=bar_colors, opacity=1.0),
            offsetgroup=1,
            showlegend=False,
            customdata=[[v, (v / total_end * 100) if total_end else 0] for v in end_existing_vals],
            hovertemplate=f'%{{x}} · {end_label} (existing)<br>%{{customdata[0]:,}} customers (%{{customdata[1]:.1f}}% of total)<extra></extra>'
        ))
        fig_size.add_trace(go.Bar(
            name=f'{end_label} (new)',
            x=all_segs,
            y=end_new_vals,
            marker=dict(color=bar_colors, opacity=1.0, pattern=dict(shape='/', size=6, solidity=0.3, fgcolor='rgba(255,255,255,0.85)')),
            offsetgroup=1,
            showlegend=False,
            base=end_existing_vals,
            customdata=[[v, (v / total_end * 100) if total_end else 0] for v in end_new_vals],
            hovertemplate=f'%{{x}} · {end_label} (new)<br>%{{customdata[0]:,}} customers (%{{customdata[1]:.1f}}% of total)<extra></extra>'
        ))

        fig_size.update_layout(
            barmode='group',
            height=320,
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis_gridcolor='#e5e7eb',
            yaxis_title='Customers',
            legend=dict(orientation='h', yanchor='top', y=0.99, xanchor='left', x=0.01, bgcolor='rgba(255,255,255,0.8)')
        )
        st.plotly_chart(fig_size, use_container_width=True)
        st.caption(f"Solid bars = {start_label} (light) vs {end_label} existing (dark). Striped = new customers acquired. Total: {total_start:,} → {total_end:,} ({total_new:,} new).")

        st.markdown("")

        # -- Chart 2: Normalised heatmap, full width below --
        st.markdown("**Where did each segment's customers go?** (% of row)")
        
        matrix_norm = migration_matrix.div(migration_matrix.sum(axis=1), axis=0) * 100
        col_labels = list(migration_matrix.columns)
        row_labels = list(migration_matrix.index)
        
        # Build z-values (normalized percentages) and hover text
        z_vals = []
        hover_text = []
        raw_counts = []
        
        for row_seg in row_labels:
            z_row = []
            hover_row = []
            raw_row = []
            row_total = int(counts_start.get(row_seg, 0))
            overall_total = counts_start.sum()
            
            for col_seg in col_labels:
                pct = matrix_norm.loc[row_seg, col_seg] if (row_seg in matrix_norm.index and col_seg in matrix_norm.columns) else 0.0
                raw = int(migration_matrix.loc[row_seg, col_seg]) if (row_seg in migration_matrix.index and col_seg in migration_matrix.columns) else 0
                
                z_row.append(pct)
                raw_row.append(raw)
                hover_row.append(
                    f"{row_seg} → {col_seg}<br>" +
                    f"{raw:,} customers<br>" +
                    f"{pct:.1f}% of row"
                )
            
            z_vals.append(z_row)
            hover_text.append(hover_row)
            raw_counts.append(raw_row)
        
        # Create heatmap with custom colorscale (light to segment color intensity)
        fig_heat = go.Figure(data=go.Heatmap(
            z=z_vals,
            x=col_labels,
            y=[f"{seg}<br>n={int(counts_start.get(seg, 0))}" for seg in row_labels],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="% of Segment", thickness=15, len=0.7),
            hovertext=hover_text,
            hoverinfo='text',
            text=[[f"{z_vals[i][j]:.0f}%<br>({raw_counts[i][j]:,})" for j in range(len(col_labels))] for i in range(len(row_labels))],
            texttemplate='%{text}'
        ))
        
        fig_heat.update_layout(
            height=250 + (len(row_labels) * 35),
            margin=dict(t=20, b=50, l=140, r=100),
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="End Segment", side='bottom', tickangle=-45),
            yaxis=dict(title="Start Segment"),
            hovermode='closest',
            font=dict(size=11, color='#1f2937')
        )
        st.plotly_chart(fig_heat, use_container_width=True, key='seg_migration_heat')
        
        st.caption(
            "Percentages show % of each starting segment that moved to each ending segment. "
            "Raw customer counts shown below percentages. "
            "New customers acquired during the period are excluded from this analysis."
        )

    else:
        st.info("Insufficient data for migration analysis.")

    st.markdown("---")

    # ---- Section 4: Segment Cohort Retention (if lifetimes available) ----
    if LIFETIMES_AVAILABLE:
        st.markdown(f"### Segment Retention Profiles (Cohorts Acquired in {lookback_months_seg}-Month Window)")
        
        rfm_seg_snapshot = calculate_rfm_as_of_date(transactions, selected_eom_seg)
        if len(rfm_seg_snapshot) > 0:
            rfm_seg_snapshot, _ = assign_segments(rfm_seg_snapshot, rfm_ref)
            segment_list = sorted([s for s in rfm_seg_snapshot['Segment'].unique() if pd.notna(s) and s not in ['Unknown']])
            
            retention_by_seg, valid_milestones = get_segment_retention_curves(transactions, rfm_ref, segment_list, selected_eom_seg, lookback_months_seg, regions=seg_regions)
            
            if retention_by_seg:
                fig_ret = go.Figure()
                
                for segment in segment_list:
                    if segment in retention_by_seg:
                        ret_values = [retention_by_seg[segment].get(m, None) for m in valid_milestones]
                        color = seg_colors_temp.get(segment, {}).get('bg', '#3b82f6')
                        fig_ret.add_trace(go.Scatter(
                            x=valid_milestones,
                            y=ret_values,
                            name=segment,
                            mode='lines+markers',
                            line=dict(color=color, width=2.5),
                            marker=dict(size=8),
                            hovertemplate=f"{segment}<br>%{{x}}<br>%{{y:.1%}} retained<extra></extra>"
                        ))
                
                fig_ret.update_layout(
                    height=350, plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=40, b=20, l=20, r=20),
                    yaxis_gridcolor='#e5e7eb', xaxis_gridcolor='#f3f4f6',
                    yaxis_tickformat='.0%',
                    hovermode='x unified',
                    yaxis_title='Retention Rate',
                    xaxis_title='Months Since Acquisition'
                )
                st.plotly_chart(fig_ret, use_container_width=True)
                st.caption(f"Retention rates for customers acquired between {analysis_start.strftime('%b %Y')} and {selected_eom_seg.strftime('%b %Y')}.")
            else:
                st.info("Insufficient cohort data for retention analysis in this window.")
        else:
            st.info("No segment data available for selected date.")



# ============================================================================
# FORECAST TAB
# ============================================================================
with tab_forecast:
    st.markdown("## Forecast")

    if not LIFETIMES_AVAILABLE:
        st.error(
            "The `lifetimes` package is not installed, so the existing-customer "
            "BG/NBD forecast can't run. Install it with `pip install lifetimes` "
            "and restart the app. New-customer forecast and cohort analysis below "
            "still work without it."
        )

    # ---- Horizon & comparison controls ----
    fh1, fh2 = st.columns([2, 1])
    with fh1:
        horizon_label = st.radio(
            "Forecast Horizon",
            options=['3 Months', '6 Months', '12 Months'],
            horizontal=True,
            key='fc_horizon'
        )
    with fh2:
        fc_comparison_type = st.radio(
            "Compare to",
            options=['Previous Period', 'Previous Year'],
            horizontal=True,
            key='fc_compare'
        )
    horizon_months = {'3 Months': 3, '6 Months': 6, '12 Months': 12}[horizon_label]

    forecast_start = (transactions['transaction_date'].max() + pd.DateOffset(months=1)).replace(day=1)
    current_end_exclusive = forecast_start + pd.DateOffset(months=horizon_months)

    # ---- vs actual comparison period (calendar-block aligned: quarter/half/year, split new/existing) ----
    comparison_revenue = calculate_actual_comparison_revenue(
        transactions, forecast_start=forecast_start, horizon_months=horizon_months, comparison_type=fc_comparison_type
    )
    st.caption(
        f"Current: {forecast_start.strftime('%d.%m.%Y')} → {(current_end_exclusive - pd.Timedelta(days=1)).strftime('%d.%m.%Y')}  |  "
        f"Comparison: {comparison_revenue['start'].strftime('%d.%m.%Y')} → {(comparison_revenue['end'] - pd.Timedelta(days=1)).strftime('%d.%m.%Y')}"
    )

    # ---- Baseline ranges from last 12 cohorts (min/max actuals, not +/-% of median) ----
    ranges = calculate_cohort_assumption_ranges(transactions, max_months_back=12, milestones=(0, 3, 6, 9, 12))
    retention_medians = ranges['retention_medians']
    retention_ranges = ranges['retention_ranges']

    # Pick the retention milestone closest to (but not exceeding) the chosen horizon,
    # used as the single "retention rate" slider reference point
    available_milestones = [m for m in retention_medians if m > 0 and not np.isnan(retention_medians[m])]
    ref_milestone = max([m for m in available_milestones if m <= horizon_months], default=min(available_milestones, default=3))
    median_retention_rate = retention_medians.get(ref_milestone, 0.3)
    retention_min, retention_max = retention_ranges.get(ref_milestone, (median_retention_rate * 0.9, median_retention_rate * 1.2))

    st.markdown("---")
    st.markdown("### Assumptions")
    st.caption(
        f"Slider ranges are the **min/max** observed across the last 12 acquisition cohorts "
        f"(Historical Cohort Analysis below). Defaults are the median. "
        f"Retention slider reflects the M{ref_milestone} milestone."
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        acq_min = int(round(ranges['acquisition_min']))
        acq_max = int(round(ranges['acquisition_max']))
        acq_default = int(round(ranges['acquisition_median']))
        acquisition_per_month = st.slider(
            "New Customers / Month",
            min_value=acq_min,
            max_value=max(acq_max, acq_min + 1),
            value=min(max(acq_default, acq_min), acq_max),
            step=1,
            key='fc_acquisition'
        )
    with s2:
        revenue_per_new_customer = st.slider(
            "Avg Revenue per New Customer (€)",
            min_value=round(ranges['revenue_min'], 2),
            max_value=round(ranges['revenue_max'], 2),
            value=round(ranges['revenue_median'], 2),
            key='fc_revenue'
        )
    with s3:
        retention_rate = st.slider(
            f"Retention Rate (M{ref_milestone})",
            min_value=round(retention_min, 3),
            max_value=round(retention_max, 3),
            value=round(median_retention_rate, 3),
            key='fc_retention'
        )

    # Scale every milestone's retention proportionally to the slider's deviation
    # from its own median, so adjusting the slider cascades through M3/M6/M9/M12
    retention_scale = (retention_rate / median_retention_rate) if median_retention_rate else 1.0
    scaled_retention_medians = {
        m: (1.0 if m == 0 else min(1.0, r * retention_scale))
        for m, r in retention_medians.items() if not np.isnan(r)
    }

    st.markdown("---")

    # ---- Build forecasts ----
    # Score RFM as of forecast start date
    rfm_at_forecast = calculate_rfm_as_of_date(transactions, forecast_start)
    rfm_at_forecast, segment_colors_forecast = assign_segments(rfm_at_forecast, rfm_ref)
    
    # Separate existing customers and new customers
    existing_cust_forecast = rfm_at_forecast[rfm_at_forecast['Segment'] != 'New Customer']
    
    # Fit BG/NBD per RFM segment and collect results
    segment_clv_dict = {}
    segment_warnings = []
    if LIFETIMES_AVAILABLE and len(existing_cust_forecast) > 0:
        for segment in sorted(existing_cust_forecast['Segment'].dropna().unique()):
            segment_data = existing_cust_forecast[existing_cust_forecast['Segment'] == segment]
            if len(segment_data) > 0:
                # Filter transactions for this segment's customers
                segment_customers = set(segment_data['customer_id'].values)
                segment_trans = transactions[transactions['customer_id'].isin(segment_customers)]
                
                # Fit BG/NBD for this segment
                bgf_seg, ggf_seg, fitter_input_seg = fit_bgnbd_gammagamma(segment_trans)
                if bgf_seg is not None:
                    clv_seg = forecast_existing_customer_clv(bgf_seg, ggf_seg, fitter_input_seg, months=horizon_months)
                else:
                    clv_seg = 0.0
                    segment_warnings.append(segment)
                segment_clv_dict[segment] = clv_seg
    
    if segment_warnings:
        warning_text = ", ".join(segment_warnings)
        st.warning(
            f"BG/NBD models could not converge for segment(s): {warning_text}. "
            f"CLV for these segments shown as €0 — new-customer "
            f"forecast and cohort analysis are unaffected."
        )
    
    existing_clv = sum(segment_clv_dict.values())

    new_cohort_forecast = build_new_customer_forecast(
        forecast_months=horizon_months,
        acquisition_per_month=acquisition_per_month,
        revenue_per_new_customer=revenue_per_new_customer,
        retention_medians=scaled_retention_medians,
        start_date=forecast_start
    )
    new_customer_clv = new_cohort_forecast['Revenue'].sum()
    total_forecast_clv = existing_clv + new_customer_clv

    # ---- Section 1: Summary KPIs ----
    st.markdown("### Summary")
    k1, k2, k3 = st.columns(3)
    with k1:
        metric_card(f"Total Forecast CLV ({horizon_label})", total_forecast_clv, comparison_revenue['total'], fmt='currency')
    with k2:
        metric_card("From Existing Customers", existing_clv, comparison_revenue['existing'], fmt='currency')
    with k3:
        metric_card("From New Customers", new_customer_clv, comparison_revenue['new'], fmt='currency')

    comp_label = 'previous period' if fc_comparison_type == 'Previous Period' else 'previous year'
    st.caption(
        f"vs actual {comp_label}: "
        f"Total €{comparison_revenue['total']:,.0f}  |  "
        f"Existing €{comparison_revenue['existing']:,.0f}  |  "
        f"New €{comparison_revenue['new']:,.0f}"
    )


    
    st.markdown("---")
    st.markdown("### Revenue Trend & Forecast by Segment")
    
    # Build historical cumulative monthly data (cached: one call covers all 12 months)
    hist_lookback_months = 12
    hist_start = forecast_start - pd.DateOffset(months=hist_lookback_months)
    hist_months_axis, hist_cumulative_data = get_historical_segment_revenue(
        transactions, rfm_ref, hist_start, forecast_start, hist_lookback_months
    )
    
    all_segments = sorted(set(segment_clv_dict.keys()) | {s for m in hist_cumulative_data.values() for s in m.keys()} - {'Unknown'})
    forecast_months_axis = [(pd.Timestamp(forecast_start) + pd.DateOffset(months=i)).strftime('%b %Y') for i in range(horizon_months)]
    
    # Cumulative line chart (efficient: vectorized calculations)
    last_hist = {s: hist_cumulative_data[hist_months_axis[-1]].get(s, 0) for s in all_segments} if hist_months_axis else {s: 0 for s in all_segments}
    # segment_clv_dict holds the TOTAL forecasted CLV for the whole horizon (not a monthly rate),
    # so split it evenly across the horizon's months to build a straight-line cumulative ramp.
    forecast_total = {s: segment_clv_dict.get(s, 0) for s in all_segments}
    forecast_monthly = {s: (v / horizon_months if horizon_months else 0) for s, v in forecast_total.items()}
    
    fig_trend = go.Figure()
    for segment in all_segments:
        if segment == 'New Customer':
            continue  # handled separately below using the real new-customer forecast, not segment_clv_dict
        hist_vals = [hist_cumulative_data.get(m, {}).get(segment, 0) for m in hist_months_axis]
        forecast_cumsum = [last_hist[segment] + forecast_monthly[segment] * (j + 1) for j in range(horizon_months)]
        bridge_x = ([hist_months_axis[-1]] if hist_months_axis else []) + forecast_months_axis
        bridge_y = ([hist_vals[-1]] if hist_vals else []) + forecast_cumsum
        color = segment_colors_forecast.get(segment, {}).get('bg', '#3b82f6')
        
        fig_trend.add_trace(go.Scatter(x=hist_months_axis, y=hist_vals, name=segment, mode='lines+markers', line=dict(color=color, width=2), marker=dict(size=4), hovertemplate=f"{segment}<br>%{{x}}<br>€%{{y:,.0f}}<extra></extra>"))
        fig_trend.add_trace(go.Scatter(x=bridge_x, y=bridge_y, name=f"{segment} (Forecast)", mode='lines', line=dict(color=color, width=2, dash='dot'), hovertemplate=f"{segment} (Forecast)<br>%{{x}}<br>€%{{y:,.0f}}<extra></extra>", showlegend=False))
    
    new_cust_monthly_vals = new_cohort_forecast.set_index('Cohort')['Revenue'].reindex(forecast_months_axis).fillna(0).values
    new_cust_last_hist = last_hist.get('New Customer', 0)
    new_cust_cumsum = np.cumsum(new_cust_monthly_vals) + new_cust_last_hist
    new_cust_bridge_x = ([hist_months_axis[-1]] if hist_months_axis else []) + forecast_months_axis
    new_cust_hist_vals = [hist_cumulative_data.get(m, {}).get('New Customer', 0) for m in hist_months_axis]
    new_cust_bridge_y = ([new_cust_hist_vals[-1]] if new_cust_hist_vals else []) + list(new_cust_cumsum)
    new_cust_color = segment_colors_forecast.get('New Customer', {}).get('bg', '#6b7280')
    if hist_months_axis:
        fig_trend.add_trace(go.Scatter(x=hist_months_axis, y=new_cust_hist_vals, name='New Customer', mode='lines+markers', line=dict(color=new_cust_color, width=2), marker=dict(size=4), hovertemplate="New Customer<br>%{x}<br>€%{y:,.0f}<extra></extra>"))
    fig_trend.add_trace(go.Scatter(x=new_cust_bridge_x, y=new_cust_bridge_y, name='New Customer (Forecast)', mode='lines+markers', line=dict(color=new_cust_color, width=2, dash='dot'), marker=dict(size=4), hovertemplate="New Customer (Forecast)<br>%{x}<br>€%{y:,.0f}<extra></extra>"))
    
    if hist_months_axis:
        fig_trend.add_shape(type="line", x0=hist_months_axis[-1], x1=hist_months_axis[-1], y0=0, y1=1, yref="paper", line=dict(color="#9ca3af", width=2, dash="dot"))
        fig_trend.add_annotation(x=hist_months_axis[-1], y=1.05, yref="paper", text="Forecast Start", showarrow=False, font=dict(size=11, color="#6b7280"))
    
    fig_trend.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=20, l=20, r=20), yaxis_gridcolor='#e5e7eb', xaxis_gridcolor='#f3f4f6', legend=dict(orientation='v', y=1, x=1.02), yaxis_title='Cumulative Revenue (€)', xaxis_title='Month', hovermode='x unified')
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("---")
    st.markdown(f"### {get_forecast_report_title(horizon_label)}")
    
    comp_type = fc_comparison_type
    actual_trans = transactions[(transactions['transaction_date'] >= comparison_revenue['start']) & (transactions['transaction_date'] < comparison_revenue['end'])]
    rfm_actual = calculate_rfm_as_of_date(transactions, comparison_revenue['end'] - pd.Timedelta(days=1))
    rfm_actual, _ = assign_segments(rfm_actual, rfm_ref)
    actual_trans = actual_trans.merge(rfm_actual[['customer_id', 'Segment']], on='customer_id', how='left')
    actual_trans['Segment'] = actual_trans['Segment'].fillna('Unknown')
    actual_by_seg = actual_trans.groupby('Segment')['amount'].sum().to_dict()
    # FIXED: Use total values (not monthly avg) to match forecast period
    actual_monthly_avg = {s: v for s, v in actual_by_seg.items()}

    cols = st.columns(3)
    for idx, segment in enumerate(all_segments):
        with cols[idx % 3]:
            if segment == 'New Customer':
                forecast_val = new_customer_clv  # not in segment_clv_dict; forecast separately via cohort model
            else:
                forecast_val = forecast_total.get(segment, 0)
            actual_val = actual_monthly_avg.get(segment, 0)
            metric_card(segment, forecast_val, actual_val, fmt='currency')


    # ---- Section 3: New customer cohort progression table ----
    st.markdown("### New Customer Cohort Progression (Forecast)")
    st.caption("Each row is a future acquisition cohort. Columns show retained customers at each milestone, capped to the selected horizon.")
    milestone_cols = [c for c in ['M0', 'M3', 'M6', 'M9', 'M12'] if c in new_cohort_forecast.columns]
    display_forecast = new_cohort_forecast[['Cohort', 'New Customers'] + milestone_cols + ['Revenue']].copy()
    # M0 stays as-is (whole number by construction); M3/M6/M9/M12 rounded to whole numbers for display
    round_cols = [c for c in milestone_cols if c != 'M0']
    for c in round_cols:
        display_forecast[c] = display_forecast[c].round(0)
    st.dataframe(
        style_cohort_heatmap(display_forecast, milestone_cols, fmt='{:,.0f}', extra_formats={'Revenue': '€{:,.0f}'}),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")

    # ---- Section 4: Historical cohort analysis (actuals, last 12 cohorts) ----
    st.markdown("### Historical Cohort Analysis (Last 12 Cohorts, Actuals)")
    cohort_value_toggle = st.radio(
        "Display",
        options=['Customer Count', 'Revenue', 'Retention Rate'],
        horizontal=True,
        key='fc_cohort_value'
    )
    hist_milestone_cols_full = ['M0', 'M3', 'M6', 'M9', 'M12']

    if cohort_value_toggle == 'Retention Rate':
        hist_counts_df, _ = build_monthly_cohort_table(
            transactions, value='customers', max_months_back=12, milestones=(0, 3, 6, 9, 12)
        )
        hist_milestone_cols = [c for c in hist_milestone_cols_full if c in hist_counts_df.columns]
        hist_rate_df = cohort_table_to_retention_rate(hist_counts_df, hist_milestone_cols)
        hist_display = hist_rate_df[['Cohort', 'New Customers'] + hist_milestone_cols]
        st.dataframe(
            style_cohort_heatmap(hist_display, hist_milestone_cols, fmt='{:.1%}'),
            use_container_width=True, hide_index=True
        )
    else:
        cohort_value_mode = 'customers' if cohort_value_toggle == 'Customer Count' else 'revenue'
        hist_cohort_df, _ = build_monthly_cohort_table(
            transactions, value=cohort_value_mode, max_months_back=12, milestones=(0, 3, 6, 9, 12)
        )
        hist_milestone_cols = [c for c in hist_milestone_cols_full if c in hist_cohort_df.columns]
        hist_display = hist_cohort_df[['Cohort', 'New Customers'] + hist_milestone_cols]
        fmt = '€{:,.0f}' if cohort_value_mode == 'revenue' else '{:,.0f}'
        st.dataframe(
            style_cohort_heatmap(hist_display, hist_milestone_cols, fmt=fmt),
            use_container_width=True, hide_index=True
        )

    st.caption(
        "Retention medians used as slider defaults: " +
        ", ".join([f"M{m} = {v*100:.1f}%" for m, v in retention_medians.items() if m > 0 and not np.isnan(v)])
    )