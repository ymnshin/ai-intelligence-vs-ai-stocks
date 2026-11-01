#!/usr/bin/env python3
"""Build the AI-intelligence frontier series from Epoch AI benchmark data.

Data source (retrieved 2026-11-01):
  Epoch AI, 'Capabilities & benchmarking', epoch.ai/benchmarks
  Download documented at https://epoch.ai/benchmarks/use-this-data
  File: https://epoch.ai/data/benchmark_data.zip  (CC BY 4.0)
  Inside: epoch_capabilities_index/eci_scores.csv  (Epoch Capabilities Index)
          mmlu_external.csv                        (MMLU scores, robustness)

Primary series: ECI (Epoch Capabilities Index) -- a composite capability
index Epoch AI fits across many benchmarks so scores are comparable over
time. Frontier(t) = max ECI over all models released on or before t.

Robustness series: MMLU frontier (running max of reported MMLU accuracy).

Note on the alternative that was tried first: Artificial Analysis
Intelligence Index. Their public models page only embeds full index values
for ~30 recently charted models, which is not enough to build a 2023-2026
frontier; the complete history needs an API key. ECI is used instead.

Outputs:
  data/eci_models.csv            model-level ECI scores used
  data/mmlu_models.csv           model-level MMLU scores used
  data/intelligence_frontier.csv daily step frontier (ECI + MMLU)
"""
import io
import urllib.request
import zipfile
from datetime import date

import pandas as pd

ZIP_URL = "https://epoch.ai/data/benchmark_data.zip"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
START = "2021-01-01"
END = "2026-10-30"


def load_zip_csv(name: str) -> pd.DataFrame:
    req = urllib.request.Request(ZIP_URL, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(name) as f:
            return pd.read_csv(f)


def main() -> None:
    today = date.today().isoformat()

    # --- ECI model-level table ---
    eci = load_zip_csv("epoch_capabilities_index/eci_scores.csv")
    eci = eci.rename(columns={"Model": "model", "date": "release_date",
                              "Organization": "organization"})
    eci["release_date"] = pd.to_datetime(eci["release_date"])
    eci = eci.dropna(subset=["eci", "release_date"]).sort_values("release_date")
    eci_out = eci[["model", "release_date", "eci", "eci_ci_low",
                   "eci_ci_high", "organization"]].copy()
    eci_out["release_date"] = eci_out["release_date"].dt.date
    eci_out["source"] = "Epoch AI, Epoch Capabilities Index (eci_scores.csv), " \
                        "via https://epoch.ai/data/benchmark_data.zip " \
                        f"(retrieved {today}), CC BY 4.0"
    eci_out.to_csv("data/eci_models.csv", index=False)
    print(f"eci_models: {len(eci_out)} models, "
          f"{eci['release_date'].min().date()} .. {eci['release_date'].max().date()}")

    # --- MMLU model-level table (robustness) ---
    mmlu = load_zip_csv("mmlu_external.csv")
    mmlu = mmlu.rename(columns={"Model version": "model", "EM": "mmlu",
                                "Release date": "release_date",
                                "Organization": "organization",
                                "Source link": "source_link"})
    mmlu["release_date"] = pd.to_datetime(mmlu["release_date"], errors="coerce")
    mmlu = mmlu.dropna(subset=["mmlu", "release_date"]).sort_values("release_date")
    mmlu_out = mmlu[["model", "release_date", "mmlu", "organization",
                     "source_link"]].copy()
    mmlu_out["release_date"] = mmlu_out["release_date"].dt.date
    mmlu_out.to_csv("data/mmlu_models.csv", index=False)
    print(f"mmlu_models: {len(mmlu_out)} models, "
          f"{mmlu['release_date'].min().date()} .. {mmlu['release_date'].max().date()}")

    # --- Daily frontier step series on a calendar grid ---
    cal = pd.date_range(START, END, freq="D")

    def frontier(models: pd.DataFrame, value: str, prefix: str) -> pd.DataFrame:
        m = models.set_index("release_date")[value].sort_index()
        m = m.groupby(level=0).max()
        daily = m.reindex(cal.union(m.index)).sort_index().ffill().cummax()
        daily = daily.reindex(cal)
        out = pd.DataFrame({f"frontier_{prefix}": daily.values}, index=cal)
        # frontier model name: last model (by date) whose value == running max
        run_max = models[value].cummax()
        steps = models[models[value] == run_max][["release_date", "model"]]
        steps = steps.drop_duplicates("release_date", keep="last")
        step_names = steps.set_index("release_date")["model"]
        name_daily = step_names.reindex(cal.union(step_names.index)) \
            .sort_index().ffill().reindex(cal)
        out[f"frontier_{prefix}_model"] = name_daily.values
        return out

    f_eci = frontier(eci.rename(columns={"release_date": "release_date"}),
                     "eci", "eci")
    f_mmlu = frontier(mmlu, "mmlu", "mmlu")
    fr = f_eci.join(f_mmlu)
    fr.index.name = "date"
    fr.to_csv("data/intelligence_frontier.csv")
    steps = fr.dropna(subset=["frontier_eci"])
    steps = steps[steps["frontier_eci"].diff().fillna(1) != 0]
    print(f"frontier saved: {len(fr)} days; ECI steps: {len(steps)}")
    for d, r in steps.iterrows():
        print(f"  {d.date()}  ECI={r['frontier_eci']:.2f}  {r['frontier_eci_model']}")


if __name__ == "__main__":
    main()
