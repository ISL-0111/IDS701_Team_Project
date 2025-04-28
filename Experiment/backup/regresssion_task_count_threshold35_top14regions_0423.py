"""
This is a quasi-experimental causal model, more specifically a threshold-based treatment-effect model.

Model Description: Threshold-Based Causal Regression 

Objective:
    Evaluate whether delivery performance is affected more significantly when a courier is under high task load.

Main Modeling Approach:
    - The model introduces a threshold effect using:
        - task_count_c: task_count_c = task_count - threshold
        - high_load: Binary indicator (1 if task_count > threshold)
        - task_count_c * high_load: Interaction term to detect if extra workload under high load worsens performance

    - Fixed effects are included for:
        - region_id (to control for regional delivery conditions)
        - hour_bin (to control for time-of-day variation)

Interpretation:
    - Baseline effect of task_count_c represents normal (low-load) days.
    - The interaction term shows: 
        "How much more does each additional task affect delivery time when the courier is already under 'high load?'"

!!! Threshold might be 25(Refer to the code below)
Beyond this point, assigning additional tasks no longer leads to shorter delivery times — indicating that the courier may have reached a productivity ceiling.
"""

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pwlf

# --- Load Data ---
df = pd.read_parquet(r"https://github.com/ISL-0111/IDS701_Team_Project/raw/refs/heads/main/Data/cleaned_data.parquet")
df["accept_hour"] = df["accept_time"].dt.hour

# --- Step 1: Filter Data for Top 14 Regions ---
top_regions = df["region_id"].value_counts().head(14).index 
df_filtered = df[df["region_id"].isin(top_regions)]

# --- Step 2: Create Daily Aggregations per Courier ---
daily_agg = (
    df_filtered.groupby(["courier_id", "accept_date"])
    .agg(
        region_id=("region_id", "first"),
        task_count=("order_id", "nunique"),
        avg_delivery_duration_min=("delivery_duration_minutes", "mean"),
        avg_distance_km=("distance_km", "mean"),
        delivery_hour_mode=("accept_hour", lambda x: x.mode()[0]),
    )
    .reset_index()
)

# --- Threshold Finder: Rolling Average Visualization ---
daily_agg_sorted = daily_agg.sort_values("task_count")
rolling_avg = daily_agg_sorted["avg_delivery_duration_min"].rolling(window=100).mean()

import statsmodels.api as sm

# Lowess smoothing
lowess = sm.nonparametric.lowess(
    daily_agg_sorted["avg_delivery_duration_min"],
    daily_agg_sorted["task_count"],
    frac=0.2  # smoothing factor (조정 가능)
)

# Plot
plt.figure(figsize=(8,6))
plt.plot(daily_agg_sorted["task_count"], daily_agg_sorted["avg_delivery_duration_min"], '.', alpha=0.3, label="Raw Data")
plt.plot(lowess[:, 0], lowess[:, 1], 'r-', label="Lowess Smoothed")
plt.xlabel("Task Count")
plt.ylabel("Avg Delivery Time (minutes)")
plt.title("Smoothed Average Delivery Time by Task Count")
plt.ylim(0, 150) # data limit to 250
plt.legend()
plt.grid()
plt.show()
## threshold 35 

# --- Piecewise Regression for Visual Threshold Estimation (optional simulated data) ---
np.random.seed(42)
task_count = np.sort(np.random.randint(1, 75, size=500))
delivery_time = 200 - 2 * task_count + np.random.normal(0, 10, size=500)
delivery_time[task_count > 35] += 1.2 * (task_count[task_count > 35] - 35)

df_piecewise = pd.DataFrame({
    "task_count": task_count,
    "avg_delivery_duration_min": delivery_time
})

x = df_piecewise["task_count"].values
y = df_piecewise["avg_delivery_duration_min"].values
my_pwlf = pwlf.PiecewiseLinFit(x, y)
breaks = my_pwlf.fit(2)

x_hat = np.linspace(min(x), max(x), 100)
y_hat = my_pwlf.predict(x_hat)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.3, label="Data")
plt.plot(x_hat, y_hat, color='red', label="Piecewise Fit")
plt.axvline(x=25, color='blue', linestyle='--', label='Manual Threshold = 25')
plt.axvline(x=35, color='green', linestyle='--', label='Alternative Threshold = 35')
plt.xlabel("Task Count")
plt.ylabel("Avg Delivery Duration (min)")
plt.title("Piecewise Linear Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Step 3: Set Fixed Threshold = 35 ---
threshold = 35
daily_agg["high_load"] = (daily_agg["task_count"] > threshold).astype(int)
daily_agg["task_count_c"] = daily_agg["task_count"] - threshold

# --- Time Binning ---
daily_agg["hour_bin"] = pd.cut(
    daily_agg["delivery_hour_mode"],
    bins=[0, 6, 12, 18, 24],
    labels=["Night", "Morning", "Afternoon", "Evening"],
    right=False
)

# --- Trim Outliers (instead of log transform) ---
q_low, q_high = daily_agg["avg_delivery_duration_min"].quantile([0.01, 0.99])
daily_agg["duration_trimmed"] = daily_agg["avg_delivery_duration_min"].clip(lower=q_low, upper=q_high)

# --- Save Cleaned Data ---
daily_agg.to_csv("daily_agg_for_ttest_14regions_t35.csv", index=False)
print("File saved successfully.")

# --- Step 4: Causal Model Without Log Transformation ---
model = smf.ols(
    formula="""
        duration_trimmed ~ task_count_c * high_load
        + avg_distance_km
        + C(region_id)
        + C(hour_bin)
    """,
    data=daily_agg
).fit(cov_type="HC3")

# --- Step 5: Summary Results ---
print(model.summary())

# --- Step 6: VIF Calculation ---
X_vif = daily_agg[["task_count_c", "high_load", "avg_distance_km"]].copy()
X_vif["interaction"] = daily_agg["task_count_c"] * daily_agg["high_load"]
X_vif = sm.add_constant(X_vif)
vif_df = pd.DataFrame({
    "feature": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print(vif_df)

# --- Step 7: Diagnostic Checks ---
plt.scatter(model.fittedvalues, model.resid, alpha=0.4)
plt.axhline(y=0, color="gray", linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual vs. Fitted (No Log)")
plt.show()

qqplot(model.resid, line="s")
plt.title("Q-Q Plot (No Log)")
plt.show()

bp_test = het_breuschpagan(model.resid, model.model.exog)
print(f"Breusch-Pagan p-value (No Log): {bp_test[1]:.4f}")


### Not yet ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sort by task_count
daily_agg_sorted = daily_agg.sort_values("task_count")

# Define bin size
bin_size = 3  # you can adjust (e.g., every 3 tasks)

# Create task_count bins
daily_agg_sorted['task_count_bin'] = (daily_agg_sorted['task_count'] // bin_size) * bin_size

# Calculate mean delivery duration by bin
bin_summary = (
    daily_agg_sorted.groupby('task_count_bin')
    .agg(avg_duration=('avg_delivery_duration_min', 'mean'), count=('avg_delivery_duration_min', 'size'))
    .reset_index()
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(bin_summary['task_count_bin'], bin_summary['avg_duration'], marker='o', linestyle='-', label="Binned Average")
plt.xlabel("Task Count (Binned)")
plt.ylabel("Avg Delivery Time (minutes)")
plt.title("Average Delivery Duration by Task Count (Binned)")
plt.grid(True)
plt.axvline(x=35, color='red', linestyle='--', label='Threshold = 35')
plt.legend()
plt.show()
