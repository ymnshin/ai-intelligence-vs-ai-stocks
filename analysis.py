#!/usr/bin/env python3
"""Test: 'as AI intelligence rises, AI stocks rise'.

Inputs : data/prices.csv, data/intelligence_frontier.csv
Outputs: results.csv, summary.json, chart_intelligence_vs_stocks.svg

Design
------
Monthly (2023-03 .. 2026-10, months with an ECI frontier reading):
  dI_t   = log change of month-end ECI frontier
  excess = monthly log return minus S&P 500 log return (NVDA, SOX, basket)
  1) Pearson / Spearman correlations: contemporaneous, intelligence leads
     by 1 month (dI_t vs excess_{t+1}), intelligence lags (dI_t vs excess_{t-1})
  2) OLS: excess_t ~ 1 + dI_t + SPX_ret_t + dUS10Y_t
  3) Surprise: residual change of log frontier around its linear time trend
  Robustness: same contemporaneous correlations with the MMLU frontier
  (MMLU reporting fades after 2024, so this is a shorter sample).

Event study (daily):
  Events = the 10 largest single-day jumps of the ECI frontier.
  Abnormal log return = asset log return - S&P 500 log return.
  CARs over [-1,+5] and [0,+20] trading days; mean / median / one-sample
  t-test / sign test across events. Day 0 = first trading day on or after
  the release date.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASKET = ["NVDA", "AVGO", "MU", "MSFT", "GOOGL", "META", "AMZN", "TSM"]
ASSETS = ["NVDA", "SOX", "BASKET"]
FRONTIER_START = "2023-02-24"


# ---------------------------------------------------------------- helpers
def ols(y: np.ndarray, X: np.ndarray, names: list[str]) -> dict:
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot
    return {"n": n, "r2": r2,
            "coef": {names[i]: {"beta": float(beta[i]), "se": float(se[i]),
                                "t": float(t[i]), "p": float(p[i])}
                     for i in range(k)}}


def corr_rows(rows, section, series, asset, timing, x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    r, pr = stats.pearsonr(x, y)
    rho, ps = stats.spearmanr(x, y)
    rows.append({"section": section, "series": series, "asset": asset,
                 "timing_or_window": timing, "statistic": "pearson_r",
                 "estimate": r, "n": len(x), "p_value": pr, "note": ""})
    rows.append({"section": section, "series": series, "asset": asset,
                 "timing_or_window": timing, "statistic": "spearman_rho",
                 "estimate": rho, "n": len(x), "p_value": ps, "note": ""})
    return {"pearson_r": float(r), "pearson_p": float(pr),
            "spearman_rho": float(rho), "spearman_p": float(ps),
            "n": int(len(x))}


# ---------------------------------------------------------------- load
prices = pd.read_csv("data/prices.csv", parse_dates=["date"], index_col="date")
frontier = pd.read_csv("data/intelligence_frontier.csv",
                       parse_dates=["date"], index_col="date")

logp = np.log(prices)
dret = logp.diff()                       # daily log returns
abn = dret.sub(dret["SPX"], axis=0)      # daily abnormal log returns

# monthly
m_price = logp.resample("M").last()
m_ret = m_price.diff()
m_front = frontier.resample("M").last()
m = pd.DataFrame({
    "dI": np.log(m_front["frontier_eci"]).diff(),
    "dI_mmlu": np.log(m_front["frontier_mmlu"]).diff(),
    "SPX_ret": m_ret["SPX"],
    "dUS10Y": prices["US10Y"].resample("M").last().diff() / 10.0,
})
for a in ["NVDA", "SOX"]:
    m[f"excess_{a}"] = m_ret[a] - m_ret["SPX"]
m["excess_BASKET"] = m_ret[BASKET].mean(axis=1) - m_ret["SPX"]

sample = m.loc["2023-03":].dropna(subset=["dI"])
print(f"monthly sample: {sample.index.min().date()} .. "
      f"{sample.index.max().date()}, n={len(sample)}")

rows: list[dict] = []
summary: dict = {"monthly_sample": {"start": str(sample.index.min().date()),
                                    "end": str(sample.index.max().date()),
                                    "n_months": int(len(sample))},
                 "correlations": {}, "regressions": {}, "surprise": {},
                 "event_study": {}}

# ---------------------------------------------------------------- 1) correlations
dI = sample["dI"].to_numpy()
for asset in ASSETS:
    ex = sample[f"excess_{asset}"].to_numpy()
    summary["correlations"][asset] = {
        "contemporaneous": corr_rows(rows, "correlation", "ECI", asset,
                                     "contemporaneous", dI, ex),
        "intelligence_leads_1m": corr_rows(rows, "correlation", "ECI", asset,
                                           "intelligence_leads_1m",
                                           dI[:-1], ex[1:]),
        "intelligence_lags_1m": corr_rows(rows, "correlation", "ECI", asset,
                                          "intelligence_lags_1m",
                                          dI[1:], ex[:-1]),
    }
# robustness: MMLU frontier (shorter sample, reporting fades after 2024)
sub = sample.dropna(subset=["dI_mmlu"])
for asset in ASSETS:
    summary["correlations"].setdefault("MMLU_robustness", {})[asset] = \
        corr_rows(rows, "correlation", "MMLU", asset, "contemporaneous",
                  sub["dI_mmlu"].to_numpy(), sub[f"excess_{asset}"].to_numpy())

# ---------------------------------------------------------------- 2) regressions
for asset in ASSETS:
    d = sample[["dI", "SPX_ret", "dUS10Y", f"excess_{asset}"]].dropna()
    X = np.column_stack([np.ones(len(d)), d["dI"], d["SPX_ret"], d["dUS10Y"]])
    res = ols(d[f"excess_{asset}"].to_numpy(), X,
              ["const", "dI", "SPX_ret", "dUS10Y"])
    summary["regressions"][asset] = res
    for nm, c in res["coef"].items():
        rows.append({"section": "regression", "series": "ECI", "asset": asset,
                     "timing_or_window": "monthly_ols", "statistic": f"coef_{nm}",
                     "estimate": c["beta"], "n": res["n"], "p_value": c["p"],
                     "note": f"se={c['se']:.4g}, t={c['t']:.3g}, R2={res['r2']:.4f}"})

# ---------------------------------------------------------------- 3) surprise decomposition
lvl = np.log(m_front["frontier_eci"]).loc["2023-02":].dropna()
t_idx = np.arange(len(lvl))
trend = ols(lvl.to_numpy(), np.column_stack([np.ones(len(lvl)), t_idx]),
            ["const", "t"])
slope = trend["coef"]["t"]["beta"]
# NOTE: for a step-function frontier, the monthly change of the
# trend-residual of the log level is exactly dI_t - slope (a constant
# shift), so Pearson correlations are identical to the raw ones by
# construction. We compute it in this closed form (avoids float jitter
# breaking ties among the many zero-change months) and report the
# equivalence as a design finding rather than an independent test.
surprise = sample["dI"] - slope
ss = pd.DataFrame({"surprise": surprise}).join(
    sample[[f"excess_{a}" for a in ASSETS]]).dropna()
summary["surprise"] = {"trend_growth_per_month_log": float(slope),
                       "note": "surprise_t = dI_t - trend slope; for a step "
                               "frontier this is a constant shift of dI, so "
                               "Pearson r equals the raw correlation by "
                               "construction (reported as a design finding)"}
for asset in ASSETS:
    summary["surprise"][asset] = corr_rows(
        rows, "surprise", "ECI", asset, "contemporaneous",
        ss["surprise"].to_numpy(), ss[f"excess_{asset}"].to_numpy())

# ---------------------------------------------------------------- 4) event study
fd = frontier["frontier_eci"].dropna()
steps = fd.diff()
steps = steps[(steps > 0)]
steps = steps.loc[steps.index > FRONTIER_START]  # exclude the initial level
events = steps.sort_values(ascending=False).head(10).sort_index()
print(f"events: {len(events)} largest frontier jumps")
tdays = prices.index
ev_records = []
for edate, jump in events.items():
    model = frontier.loc[edate, "frontier_eci_model"]
    i0 = int(tdays.searchsorted(edate))
    rec = {"event_date": str(edate.date()), "model": model,
           "eci_jump": float(jump), "day0": str(tdays[i0].date())}
    for asset in ["NVDA", "SOX"]:
        a = abn[asset]
        for wname, (lo, hi) in {"[-1,+5]": (-1, 5), "[0,+20]": (0, 20)}.items():
            seg = a.iloc[i0 + lo: i0 + hi + 1]
            rec[f"{asset}_{wname}"] = float(seg.sum()) if len(seg) == hi - lo + 1 else np.nan
    ev_records.append(rec)
ev = pd.DataFrame(ev_records)
ev.to_csv("data/event_list.csv", index=False)
for asset in ["NVDA", "SOX"]:
    summary["event_study"][asset] = {}
    for wname in ["[-1,+5]", "[0,+20]"]:
        car = ev[f"{asset}_{wname}"].dropna().to_numpy()
        tstat, tp = stats.ttest_1samp(car, 0.0)
        k_pos = int((car > 0).sum())
        sign_p = stats.binomtest(k_pos, len(car), 0.5).pvalue
        summary["event_study"][asset][wname] = {
            "n_events": int(len(car)), "mean_car": float(car.mean()),
            "median_car": float(np.median(car)), "t_stat": float(tstat),
            "t_p": float(tp), "n_positive": k_pos, "sign_p": float(sign_p)}
        for stat_name, val, p in [("mean_car", car.mean(), tp),
                                  ("median_car", np.median(car), np.nan),
                                  ("sign_test_p", sign_p, sign_p)]:
            rows.append({"section": "event_study", "series": "ECI",
                         "asset": asset, "timing_or_window": wname,
                         "statistic": stat_name, "estimate": float(val),
                         "n": int(len(car)), "p_value": float(p),
                         "note": f"mean_car t={tstat:.3g}; positives={k_pos}/{len(car)}"})
summary["event_study"]["events"] = ev_records

# ---------------------------------------------------------------- outputs
res_df = pd.DataFrame(rows)
res_df.to_csv("results.csv", index=False)
with open("summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- chart
fc = frontier.loc[FRONTIER_START:"2026-10-30", "frontier_eci"].dropna()
fc = fc.reindex(tdays[tdays >= FRONTIER_START]).ffill()
cum_nvda = abn["NVDA"].loc[FRONTIER_START:].cumsum() * 100
cum_sox = abn["SOX"].loc[FRONTIER_START:].cumsum() * 100
fig, ax1 = plt.subplots(figsize=(11, 5.2))
ax1.step(fc.index, fc.values, where="post", color="#444444", lw=1.6,
         label="AI intelligence frontier (ECI, left)")
ax1.set_ylabel("Epoch Capabilities Index (frontier)")
ax1.set_ylim(fc.min() * 0.92, fc.max() * 1.03)
ax2 = ax1.twinx()
ax2.plot(cum_nvda.index, cum_nvda.values, color="#76b900", lw=1.3,
         label="NVDA cumulative excess return vs S&P 500 (%, right)")
ax2.plot(cum_sox.index, cum_sox.values, color="#1f77b4", lw=1.3,
         label="SOX cumulative excess return vs S&P 500 (%, right)")
ax2.axhline(0, color="#999999", lw=0.7)
ax2.set_ylabel("Cumulative excess log return (%)")
ax1.set_title("AI intelligence frontier vs AI-stock excess returns\n"
              "ECI: Epoch AI (epoch.ai/benchmarks); prices: Yahoo Finance chart API")
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("chart_intelligence_vs_stocks.svg", format="svg")
print("saved results.csv, summary.json, chart_intelligence_vs_stocks.svg")
print(json.dumps(summary["correlations"], indent=1)[:1200])
