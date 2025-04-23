import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

# --- Load Data ---
df = pd.read_parquet("/Users/ilseoplee/IDS701_Team_Project/Data/cleaned_data.parquet")
df["accept_hour"] = df["accept_time"].dt.hour

# --- Step 1: Filter Data for Top Regions ---
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

# --- Step 3: Create relevant variables with fixed threshold = 16 ---
threshold = 16
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

# --- Step 4: Final Regression with fixed threshold ---
model = smf.ols(
    formula="""
        log_duration ~ task_count_c * high_load
        + avg_distance_km
        + C(region_id)
        + C(hour_bin)
    """,
    data=daily_agg
).fit(cov_type="HC3")

# --- Step 5: Results ---
print(model.summary())

# --- Step 6: VIF Check ---
X_vif = daily_agg[["task_count_c", "high_load", "avg_distance_km"]].copy()
X_vif["interaction"] = daily_agg["task_count_c"] * daily_agg["high_load"]
X_vif = sm.add_constant(X_vif)
vif_df = pd.DataFrame({
    "feature": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print(vif_df)

# --- Step 7: Diagnostic Plots ---
# Residual plot
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
print(f"Breusch-Pagan p-value: {bp_test[1]:.4f}")  # < 0.05 implies heteroscedasticity




##### Threshold Search #####

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

results = []
for threshold in range(5, 41):
    daily_agg["high_load"] = (daily_agg["task_count"] > threshold).astype(int)
    daily_agg["task_count_c"] = daily_agg["task_count"] - threshold

    model = smf.ols(
        formula="""
            log_duration ~ task_count_c * high_load
            + avg_distance_km
            + C(region_id)
            + C(hour_bin)
        """,
        data=daily_agg
    ).fit()

    results.append({
        "threshold": threshold,
        "R_squared": model.rsquared,
        "AIC": model.aic,
        "interaction_pval": model.pvalues.get("task_count_c:high_load", np.nan)
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values("interaction_pval").head(10)) 