import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("cleaned_data.parquet")

# ________________________________________________________________________

a=df['region_id'].value_counts().sort_values(ascending=False).reset_index()
a["region_id"] = a["region_id"].astype("category")

a = a.sort_values(by="count", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(a["region_id"].astype(str), a["count"], color='mediumseagreen')
plt.xlabel("Region ID (Categorical)")
plt.ylabel("Order Count")
plt.title("Order Count per Region (Sorted Descending)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ________________________________________________________________________


# Will consider only the top n regions (HAS BEEN ARBITRARILY ASSIGNED FOR NOW)
n_regions_under_consideration=5
regions_under_consideration=df['region_id'].value_counts().sort_values(ascending=False).head(n_regions_under_consideration).index
df_only_regions=df[df['region_id'].isin(regions_under_consideration)]

#Filtering out columns
df_only_regions=df_only_regions[["order_id","region_id","courier_id","delivery_duration_minutes","delivery_under_30","accept_date","distance_km"]]

# Number of orders per driver per day
# driver_daily_delivery_info = df_only_regions.groupby(["courier_id", "accept_date"]).agg(
#     num_unique_orders=("order_id", "nunique"),
#     total_delivery_minutes=("delivery_duration_minutes", "sum"),
#     avg_delivery_minutes=("delivery_duration_minutes", "mean")
# ).reset_index()


# Average time per order overall
driver_overall_delivery_info = df_only_regions.groupby(["courier_id"]).agg(num_unique_days=("accept_date", "nunique"),
    num_unique_orders=("order_id", "nunique"),
    total_distance=("distance_km","sum"),
    total_delivery_minutes=("delivery_duration_minutes", "sum"),
    avg_delivery_time_per_package_mins=("delivery_duration_minutes", "mean")
).reset_index()
driver_overall_delivery_info["avg_no_orders_per_day"]=driver_overall_delivery_info["num_unique_orders"]/driver_overall_delivery_info["num_unique_days"]
driver_overall_delivery_info["avg_distance_per_day"]=driver_overall_delivery_info["total_distance"]/driver_overall_delivery_info["num_unique_days"]
driver_overall_delivery_info["avg_distance_per_order"]=driver_overall_delivery_info["total_distance"]/driver_overall_delivery_info["num_unique_orders"]
driver_overall_delivery_info=driver_overall_delivery_info[["courier_id","avg_delivery_time_per_package_mins","avg_no_orders_per_day","avg_distance_per_order","avg_distance_per_day"]]


driver_overall_delivery_info=driver_overall_delivery_info.sort_values(by="avg_no_orders_per_day")
plt.scatter(driver_overall_delivery_info["courier_id"].astype(str),driver_overall_delivery_info["avg_no_orders_per_day"])
plt.show()

driver_overall_delivery_info["avg_no_orders_per_day"].describe()
driver_overall_delivery_info["avg_no_orders_per_day"].median()
driver_overall_delivery_info["avg_no_orders_per_day"].mean()

# ________________________________________________________________________

# ------- this may work with avg_distance_per_order
# Split into Treatment/Control
driver_overall_delivery_info["treatment"] = (driver_overall_delivery_info["avg_no_orders_per_day"] >= driver_overall_delivery_info["avg_no_orders_per_day"].mean()).astype(int)
driver_overall_delivery_info.to_csv("treatment_control_split.csv")

treatment=driver_overall_delivery_info[driver_overall_delivery_info["treatment"]==1]
control=driver_overall_delivery_info[driver_overall_delivery_info["treatment"]==0]

treatment["avg_delivery_time_per_package_mins"].mean()
control["avg_delivery_time_per_package_mins"].mean()

from scipy.stats import ttest_ind

t_stat, p_val = ttest_ind(treatment["avg_delivery_time_per_package_mins"], control["avg_delivery_time_per_package_mins"], equal_var=False)  # Welch’s t-test is safer

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")