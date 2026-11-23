# Retail Customer Analytics Dashboard

> Personal project: a Streamlit dashboard built on a sample retail transactions dataset.

An interactive dashboard that turns raw transaction data into a view of customer value, retention and revenue forecasts.

## What it does

- **Financial:** revenue and customer KPIs compared with the previous period or the previous year, filterable by region, category and segment.
- **Forecast:** revenue forecast vs. actuals over 3, 6 or 12 months, using customer lifetime value models (BG/NBD and Gamma-Gamma, via the `lifetimes` library).
- **RFM analysis:** segments customers by recency, frequency and monetary value. The segment rules are driven by a configurable template in the Excel source file.
- **Segment analysis:** monthly revenue and customer trends by segment, plus cohort retention curves.

## Run it

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

The source data is `challenge1_retail_churn - Datasource.xlsm`.

## Built with

Python, pandas, NumPy, Plotly, Streamlit, lifetimes, and Excel.
