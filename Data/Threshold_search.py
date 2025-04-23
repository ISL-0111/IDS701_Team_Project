import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# --- Load and preprocess your daily_agg dataset first ---
# Assumes daily_agg is already created with all necessary columns:
# ['task_count', 'avg_delivery_duration_min', 'region_id', 'avg_distance_km', 'hour_bin']

# Preprocessing
daily_agg["log_duration"] = np.log(daily_agg["avg_delivery_duration_min"].replace(0, 0.001))
q_low, q_high = daily_agg["log_duration"].quantile([0.01, 0.99])
daily_agg["log_duration"] = daily_agg["log_duration"].clip(lower=q_low, upper=q_high)

# Sweep thresholds
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
print(results_df.sort_values("interaction_pval").head())  # 유의한 변화점을 상위순으로 보기
