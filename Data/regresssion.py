import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

df = pd.read_parquet(
    r"C:\Users\DELL\UDS Project\IDS701_Team_Project\Data\cleaned_data.parquet"
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

# --- Step 3: Create 'high_load' Flag ---
# Define high_load as days with task_count > median daily tasks across all couriers
median_daily_tasks = daily_agg["task_count"].median()
daily_agg["high_load"] = (daily_agg["task_count"] > median_daily_tasks).astype(int)

# --- Step 4: Run Regression ---
model = smf.ols(
    formula="np.log(avg_delivery_duration_min) ~ task_count * high_load + avg_distance_km + C(delivery_hour_mode)",
    data=daily_agg,
).fit(cov_type="HC3")

# --- Step 5: Print Results ---
print(model.summary())

# --- Diagnostic Checks ---
# Check for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = daily_agg[["task_count", "high_load", "avg_distance_km"]]  # Exclude categorical
X["interaction"] = X["task_count"] * X["high_load"]  # Add interaction term
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
