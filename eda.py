"""Exploratory analysis: prints the data-quality findings and saves figures.

Run:  python eda.py
Writes: figures/*.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import freight

Path("figures").mkdir(exist_ok=True)
raw = freight.load("data/train_test.csv")
val = freight.load("data/validation.csv")

print("=== Data quality ===")
print(f"Rows: train_test {len(raw):,} (Jan-Oct 2025), validation {len(val):,} (Nov-Dec 2025)")
print(f"Duplicate load_ids: {raw['load_id'].duplicated().sum()}")
print(f"Negative weights: train {(raw['weight'] < 0).sum()}, validation {(val['weight'] < 0).sum()}")
print(f"Missing weights: train {raw['weight'].isna().sum()}, validation {val['weight'].isna().sum()}")
print(f"Missing market_index: train {raw['market_index'].isna().sum()}, validation {val['market_index'].isna().sum()}")
known = set(raw["pickup"]) | set(raw["delivery"])
new_cities = sorted((set(val["pickup"]) | set(val["delivery"])) - known)
unseen = (~val["pickup"].isin(known) | ~val["delivery"].isin(known)).mean()
print(f"Cities in validation but never in training: {len(new_cities)} ({', '.join(new_cities)})")
print(f"Validation rows touching an unseen city: {unseen:.1%}")

df = freight.clean(raw)
bad = freight.flag_outliers(df)
huber = freight.fit(df[~bad])
ratio = df["posted_rate"] / freight.predict(huber, df)
print(f"Corrupted rate labels: {bad.sum()} ({bad.mean():.1%}); "
      f"low cluster {ratio[ratio < 0.6].min():.2f}-{ratio[ratio < 0.6].max():.2f}x, "
      f"high cluster {ratio[ratio > 1.6].min():.2f}-{ratio[ratio > 1.6].max():.2f}x expected")
print(f"Clean rows within +/-5% of expected: {((ratio[~bad] - 1).abs() < 0.05).mean():.1%}")
print(f"Correlation of rate with distance: {df['posted_rate'].corr(df['distance']):.2f}")

# Figure 1: rate vs distance, corrupted labels highlighted
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
colors = {"Dry Van": "#2a6f97", "Reefer": "#d1495b", "Flatbed": "#edae49"}
for eq, colour in colors.items():
    part = df[(df["equipment"] == eq) & ~bad]
    ax.scatter(part["distance"], part["posted_rate"], s=2, alpha=0.25, color=colour, label=eq)
ax.scatter(df.loc[bad, "distance"], df.loc[bad, "posted_rate"], s=8, color="black",
           marker="x", label=f"Corrupted label ({bad.sum()})")
ax.set(xlabel="Distance (miles)", ylabel="Posted rate ($)", title="Rate is driven by distance; 1.4% of labels are corrupted")
ax.legend(markerscale=4, frameon=False)
fig.tight_layout(); fig.savefig("figures/rate_vs_distance.png"); plt.close(fig)

# Figure 2: actual / expected rate, showing the gap between real and corrupted
fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
ax.hist(np.log10(ratio), bins=200, color="#2a6f97")
ax.set_yscale("log")
for x in (freight.OUTLIER_LOW, freight.OUTLIER_HIGH):
    ax.axvline(np.log10(x), color="#d1495b", linestyle="--")
ticks = [0.2, 0.3, 0.5, 0.6, 1, 1.6, 2, 3, 5]
ax.set_xticks(np.log10(ticks), [f"{t}x" for t in ticks])
ax.set(xlabel="Posted rate / expected rate (log scale)", ylabel="Loads (log scale)",
       title="Corrupted labels sit at about 0.2-0.45x or 2.2-5x expected; cut-offs at 0.6x and 1.6x")
fig.tight_layout(); fig.savefig("figures/outlier_ratio.png"); plt.close(fig)

# Figure 3: market_index vs the actual monthly rate level
f = [c for c in freight.FEATURES if c != "day_of_week"]
level_model = freight.fit(df[~bad], f)
resid = np.log(df.loc[~bad, "posted_rate"]) - np.log(freight.predict(level_model, df[~bad], f))
month = df.loc[~bad, "date"].dt.to_period("M")
level = (resid.groupby(month).mean() * 100)
mi = df.groupby(df["date"].dt.to_period("M"))["market_index"].mean()
fig, ax1 = plt.subplots(figsize=(9, 4), dpi=150)
x = [p.strftime("%b") for p in level.index]
ax1.bar(x, level.values, color="#2a6f97", alpha=0.7, label="Rate level vs average (%)")
ax1.set_ylabel("Rate level vs average (%)")
ax2 = ax1.twinx()
ax2.plot(x, mi.values, color="#d1495b", marker="o", label="market_index")
ax2.set_ylabel("market_index (monthly mean)")
ax1.set_title("market_index swings 0.89-1.30 while rates move only about 2%, and the link breaks in Sep-Oct")
fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.88), frameon=False)
fig.tight_layout(); fig.savefig("figures/market_index_vs_level.png"); plt.close(fig)

# Day-of-week effect (clean data, after distance/equipment/etc.)
dow = (resid.groupby(df.loc[~bad, "date"].dt.dayofweek).mean() * 100).round(2)
dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print("Day-of-week rate effect (%):", dow.to_dict())
print("Saved figures/rate_vs_distance.png, figures/outlier_ratio.png, figures/market_index_vs_level.png")
