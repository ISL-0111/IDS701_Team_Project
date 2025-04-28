"""
This is a quasi-experimental causal model, more specifically a threshold-based treatment-effect model.

Model Description: Threshold-Based Causal Regression 

Objective:
    Evaluate whether delivery performance (log_duration) is affected more significantly when a courier is under high task load.

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
import matplotlib.pyplot as plt

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

### Threshold finder  35 ###
# --- Threshold Finder: Rolling Average Visualization ---
daily_agg_sorted = daily_agg.sort_values("task_count")
rolling_avg = daily_agg_sorted["avg_delivery_duration_min"].rolling(window=100).mean()

import statsmodels.api as sm

# Lowess smoothing
lowess = sm.nonparametric.lowess(
    daily_agg_sorted["avg_delivery_duration_min"],
    daily_agg_sorted["task_count"],
    frac=0.2  # smoothing factor
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

## Piecewise Regression ##
# Simulate data structure similar to `daily_agg` for piecewise regression example
# Since we don't have the real data, we'll create a representative version based on your code
# --- Import necessary packages ---
import numpy as np
import pandas as pd
import pwlf
import matplotlib.pyplot as plt

# --- Simulate Example Data (similar to daily_agg structure) ---
np.random.seed(42)

# Generate sorted task_count values
task_count = np.sort(np.random.randint(1, 75, size=500))

# Simulate delivery times: base downward trend + noise
delivery_time = 200 - 2 * task_count + np.random.normal(0, 10, size=500)

# Manually simulate a worsening slope after task_count > 35
worsening_start = 35
delivery_time[task_count > worsening_start] += 1.5 * (task_count[task_count > worsening_start] - worsening_start)

# Create a DataFrame
df_piecewise = pd.DataFrame({
    "task_count": task_count,
    "avg_delivery_duration_min": delivery_time
})

# --- Prepare Data for Piecewise Regression ---
x = df_piecewise["task_count"].values
y = df_piecewise["avg_delivery_duration_min"].values

# --- Fit Piecewise Linear Model (2 Segments) ---
my_pwlf = pwlf.PiecewiseLinFit(x, y)

# Fit model with 2 line segments (meaning 1 breakpoint)
breakpoints = my_pwlf.fit(2)

# Print breakpoints found
print(f"Estimated Breakpoints: {breakpoints}")

# --- Predict and Plot ---
x_hat = np.linspace(min(x), max(x), 100)
y_hat = my_pwlf.predict(x_hat)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.3, label="Observed Data")
plt.plot(x_hat, y_hat, color='red', label="Piecewise Linear Fit (2 Segments)")
plt.axvline(x=breakpoints[1], color='orange', linestyle='--', label=f"Estimated Breakpoint = {breakpoints[1]:.2f}")
plt.axvline(x=worsening_start, color='blue', linestyle='--', label=f"Hypothesis Threshold = {worsening_start}")
plt.xlabel("Task Count")
plt.ylabel("Avg Delivery Duration (min)")
plt.title("Piecewise Linear Regression: Threshold Validation")
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

# --- Log transform + trim outliers ---
daily_agg["log_duration"] = np.log(daily_agg["avg_delivery_duration_min"].replace(0, 0.001))
q_low, q_high = daily_agg["log_duration"].quantile([0.01, 0.99])
daily_agg["log_duration"] = daily_agg["log_duration"].clip(lower=q_low, upper=q_high)
# daily_agg.columns
daily_agg.to_csv("daily_agg_for_ttest_14regions_log_threshold_35.csv", index=False)
print("File saved successfully.")

# --- Step 4: Causal Model
model = smf.ols(
    formula="""
        log_duration ~ task_count_c * high_load
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
# Residual Plot
plt.scatter(model.fittedvalues, model.resid, alpha=0.4)
plt.axhline(y=0, color="gray", linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual vs. Fitted")
plt.show()

# Q-Q Plot
qqplot(model.resid, line="s")
plt.title("Q-Q Plot")
plt.show()

# Breusch-Pagan Test
bp_test = het_breuschpagan(model.resid, model.model.exog)
print(f"Breusch-Pagan p-value: {bp_test[1]:.4f}")