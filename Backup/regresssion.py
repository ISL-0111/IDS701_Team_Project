import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import het_breuschpagan
import numpy as np

df = pd.read_parquet(
    r"https://github.com/ISL-0111/IDS701_Team_Project/raw/refs/heads/main/Data/cleaned_data.parquet"
)
df["accept_hour"] = df["accept_time"].dt.hour
# --- Step 1: Filter Data for Top Regions ---
n_regions_under_consideration = 14
top_regions = df["region_id"].value_counts().head(n_regions_under_consideration).index
df_filtered = df[df["region_id"].isin(top_regions)]

# Keep only necessary columns
df_filtered = df_filtered[
    [
        "order_id",
        "region_id",
        "courier_id",
        "delivery_duration_minutes",
        "accept_date",
        "distance_km",
        "accept_hour",
    ]
]

# --- Step 2: Create Daily Aggregations per Courier ---
daily_agg = (
    df_filtered.groupby(["courier_id", "accept_date"])
    .agg(
        task_count=("order_id", "nunique"),  # Number of tasks per day
        avg_delivery_duration_min=("delivery_duration_minutes", "mean"),
        avg_distance_km=("distance_km", "mean"),
        delivery_hour_mode=(
            "accept_hour",
            lambda x: x.mode()[0],
        ),  # Most frequent hour that day
    )
    .reset_index()
)

# --- Step 3: Create relevant variables ---
# Define high_load as days with task_count > median daily tasks across all couriers
median_daily_tasks = daily_agg["task_count"].median()
daily_agg["high_load"] = (daily_agg["task_count"] > median_daily_tasks).astype(int)
# categorize the times
daily_agg["hour_bin"] = pd.cut(
    daily_agg["delivery_hour_mode"],
    bins=[0, 6, 12, 18, 24],
    labels=["Night", "Morning", "Afternoon", "Evening"],
)
# transform delivery duration to log and replace zeros with a small value (0.001) + remove outliers
daily_agg["log_duration"] = np.log(
    daily_agg["avg_delivery_duration_min"].replace(0, 0.001)
)
lower = daily_agg["log_duration"].quantile(0.01)
upper = daily_agg["log_duration"].quantile(0.99)
daily_agg["log_duration"] = daily_agg["log_duration"].clip(lower, upper)
# standarize numeric variables
# scaler = StandardScaler()
# daily_agg[["task_count", "avg_distance_km"]] = scaler.fit_transform(
#     daily_agg[["task_count", "avg_distance_km"]]
# )

# --- Step 4: Run Regression ---
model = smf.ols(
    formula="log_duration ~ task_count + np.power(task_count, 2) + avg_distance_km + task_count * avg_distance_km + C(hour_bin)",
    data=daily_agg,
).fit(cov_type="HC3")

# --- Step 5: Print Results ---
print(model.summary())

# --- Diagnostic Checks ---
# Check for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = daily_agg[["task_count", "high_load", "avg_distance_km"]]
X["interaction"] = X["task_count"] * X["avg_distance_km"]  # Add interaction term
X = sm.add_constant(X)
print(
    pd.DataFrame(
        {
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            "feature": X.columns,
        }
    )
)

# Check residuals
plt.scatter(model.fittedvalues, model.resid)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# qq plot
qqplot(model.resid, line="s")
plt.show()

# Breusch-Pagan Test for heteroscedasticity
bp_test = het_breuschpagan(model.resid, model.model.exog)
print("P-value:", bp_test[1])  # <0.05 confirms heteroscedasticity





