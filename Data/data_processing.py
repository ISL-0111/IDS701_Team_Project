import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load parquet file
df = pd.read_parquet("delivery_sh-00000-of-00001-ad9a4b1d79823540.parquet")  # Shanghai dataset

"""
    LaDe is a publicly available last-mile delivery dataset with millions of packages from industry. 
    It has three unique characteristics: 
    (1) Large-scale. It involves 10,677k packages of 21k couriers over 6 months of real-world operation.
    (2) Comprehensive information, it offers original package information, 
    such as its location and time requirements, as well as task-event information, 
    which records when and where the courier is while events such as task-accept and task-finish events happen. 
    (3) Diversity: the dataset includes data from various scenarios, such as package pick-up and delivery, and from multiple cities, each with its unique spatio-temporal patterns due to their distinct characteristics such as populations.
    ## In this case we are using Shanghai dataset 1.4 million(2023, 6 months) observations
"""
df.tail(20)
# df["courier_id"].nunique()  # 1.7k unique couriers
# df["aoi_id"].nunique()  # 1.7k unique couriers
# df.dtypes # datatype
# df.columns
# df.shape # MM1.4 observations, 17 variables


"""
    0. Data preprocessing
"""

# Haversine Distance between pickup and delivery points
from geopy.distance import geodesic

def haversine_distance(row):
    try:
        start = (row["accept_gps_lat"], row["accept_gps_lng"])
        end = (row["delivery_gps_lat"], row["delivery_gps_lng"])
        if pd.notna(start[0]) and pd.notna(start[1]) and pd.notna(end[0]) and pd.notna(end[1]):
            return geodesic(start, end).km
        else:
            return np.nan
    except:
        return np.nan

df["distance_km"] = df.apply(haversine_distance, axis=1)
df.head(5)
# Data Formatting

# formatting time
df["accept_time"] = pd.to_datetime(
    df["accept_time"], format="%m-%d %H:%M:%S", errors="coerce"
)
df["delivery_time"] = pd.to_datetime(
    df["delivery_time"], format="%m-%d %H:%M:%S", errors="coerce"
)
df["accept_gps_time"] = pd.to_datetime(
    df["accept_gps_time"], format="%m-%d %H:%M:%S", errors="coerce"
)
df["delivery_gps_time"] = pd.to_datetime(
    df["delivery_gps_time"], format="%m-%d %H:%M:%S", errors="coerce"
)

# allocating 2023 year to the time
df["accept_time"] = df["accept_time"].apply(
    lambda x: x.replace(year=2023) if pd.notna(x) else x
)
df["delivery_time"] = df["delivery_time"].apply(
    lambda x: x.replace(year=2023) if pd.notna(x) else x
)
df["accept_gps_time"] = df["accept_gps_time"].apply(
    lambda x: x.replace(year=2023) if pd.notna(x) else x
)
df["delivery_gps_time"] = df["delivery_gps_time"].apply(
    lambda x: x.replace(year=2023) if pd.notna(x) else x
)

print(
    df[["accept_time", "delivery_time", "accept_gps_time", "delivery_gps_time"]].head()
)

# Date extraction
df["accept_date"] = df["accept_time"].dt.date
df_sample = df.head(200)



"""
    1. Delivery time distribution analysis
"""

# Delivery Time Analysis
df["delivery_duration"] = df["delivery_time"] - df["accept_time"]
df["delivery_duration_minutes"] = df["delivery_duration"].dt.total_seconds() / 60
df[
    "delivery_duration_minutes"
].describe()  # Max is 47,739 minutes!? = 33days? I guess it's an outlier

# # Remove outliers above 99% percentile
q99 = df["delivery_duration_minutes"].quantile(0.99)
df_filtered = df[df["delivery_duration_minutes"] <= q99]

# Plot the histogram of filtered data
plt.figure(figsize=(10, 6))
plt.hist(
    df_filtered["delivery_duration_minutes"].dropna(),
    bins=100,
    color="midnightblue",
    edgecolor="white",
)
plt.xlabel("Delivery Duration (minutes)")
plt.ylabel("Frequency")
plt.title("Distribution of Delivery Time (After Removing Outliers)")
plt.grid(True)
plt.show()


# Summary statistics
df_filtered["delivery_duration_minutes"].describe()


"""
    2. Delivery ratio distribution analysis (30mins)
"""

# Calculate the ratio of deliveries completed within 30 minutes
deliveries_completed = df_filtered["delivery_duration_minutes"].count()
deliveries_under_30 = (df["delivery_duration_minutes"] < 30).sum()
delivery_within_30_ratio = (df["delivery_duration_minutes"] < 30).mean()

labels = ["Under 30 min", "Over 30 min"]
sizes = [deliveries_under_30, deliveries_completed - deliveries_under_30]
colors = ["blue", "orange"]

# Plot the pie chart
plt.figure(figsize=(7, 7))
plt.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=0,
    wedgeprops={"edgecolor": "white"},
)
plt.title("Delivery Time Distribution (Under 30 min vs. Over 30 min)")
plt.show()

print(f"Number of deliveries completed within 30 minutes: {deliveries_under_30}")
print(f"Number of deliveries completed: {deliveries_completed}")
print(
    f"Ratio of deliveries completed within 30 minutes: {delivery_within_30_ratio:.2%}"
)

"""
    2.1 Delivery ratio distribution analysis (various)
"""
total_count = df_filtered.shape[0] 
thresholds = [30, 45, 60, 90]
for t in thresholds:
    count = df_filtered[df_filtered["delivery_duration_minutes"] <= t].shape[0]
    percentage = (count / total_count) * 100
    print(f"{t}분 이하 배달 건수 비율: {percentage:.2f}%")


"""
    3. Proportion of Deliveries Completed Within 30 Minutes by Region
"""

# Dataset Preparation
df_filtered_3 = df_filtered.copy()
region_observation_count = df_filtered_3["region_id"].value_counts()
valid_regions = region_observation_count[
    region_observation_count > 10
].index  # regions having over 10 obersevations only
df_filtered_3 = df_filtered_3[df_filtered_3["region_id"].isin(valid_regions)]

# Calculate the ratio of deliveries completed within 30 minutes for each region, Group by region and count True/False occurrences
df_filtered_3["delivery_under_30"] = df_filtered_3["delivery_duration_minutes"] < 30
region_delivery_distribution = (
    df_filtered_3.groupby("region_id")["delivery_under_30"]
    .value_counts()
    .unstack(fill_value=0)
)

region_delivery_distribution.columns = [
    "delivery_under_30_False",
    "delivery_under_30_True",
]

# Calculate total deliveries and ratio of deliveries under 30 minutes
region_delivery_distribution["delivery_total"] = (
    region_delivery_distribution["delivery_under_30_True"]
    + region_delivery_distribution["delivery_under_30_False"]
)
region_delivery_distribution["delivery_under_30_ratio"] = (
    region_delivery_distribution["delivery_under_30_True"]
    / region_delivery_distribution["delivery_total"]
)

# Reset index and convert region_id to string
region_delivery_distribution.reset_index(inplace=True)
region_delivery_distribution["region_id"] = region_delivery_distribution[
    "region_id"
].astype(str)
region_delivery_distribution = region_delivery_distribution.sort_values(
    by="delivery_under_30_ratio", ascending=False
)

plt.figure(figsize=(12, 6))
plt.bar(
    region_delivery_distribution["region_id"],
    region_delivery_distribution["delivery_under_30_ratio"],
    color="midnightblue",
)

plt.xlabel("Region ID (Categorical)")
plt.ylabel("Proportion of Deliveries < 30 min")
plt.title(
    "Proportion of Deliveries Completed Within 30 Minutes by Region (Excluding Regions < 10 Observations)"
)

plt.xticks(rotation=90, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


"""
    3.1 Proportion of Deliveries Completed Within 30 Minutes by Region
"""
import matplotlib.pyplot as plt

# Plot the bar chart for delivery_under_30_ratio
plt.figure(figsize=(12, 6))
plt.bar(
    region_delivery_distribution["region_id"],
    region_delivery_distribution["delivery_under_30_ratio"],
    color="midnightblue",
)

plt.xlabel("Region ID (Categorical)")
plt.ylabel("Proportion of Deliveries < 30 min")
plt.title("Proportion of Deliveries Completed Within 30 Minutes by Region")
plt.xticks(rotation=90, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot (for VS Code)
plt.show()


"""
    3.2 Number of Deliveries Completed Within 30 Minutes by Region
"""
labels = region_delivery_distribution["region_id"]
under_30 = region_delivery_distribution["delivery_under_30_True"]
over_30 = region_delivery_distribution["delivery_under_30_False"]

plt.figure(figsize=(12, 6))
plt.bar(labels, under_30, label="Under 30 min", color="midnightblue")
plt.bar(labels, over_30, bottom=under_30, label="30 min or more", color="orange")

plt.xlabel("Region ID")
plt.ylabel("Number of Deliveries")
plt.title("Cumulative Histogram of Delivery Times by Region")
plt.xticks(rotation=90, ha="right")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


"""
    4. Delivery counts by region (Bar)
"""
# Sorting the data based on total deliveries in descending order
region_delivery_distribution_sorted = region_delivery_distribution.sort_values(
    by="delivery_total", ascending=False
)

# Plotting the bar chart for total deliveries by region with correct labeling
plt.figure(figsize=(12, 6))
plt.bar(
    region_delivery_distribution_sorted["region_id"],
    region_delivery_distribution_sorted["delivery_total"],
    color="midnightblue",
)

plt.xlabel("Region ID (Categorical)")
plt.ylabel("Total Deliveries")
plt.title("Total Deliveries by Region (Sorted in Descending Order)")
plt.xticks(rotation=90, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()

"""
    4. Delivery counts by region (Pie)
"""
# Preparing data for pie chart: Top 7 regions and Others
top_7 = region_delivery_distribution_sorted.head(7)
others = region_delivery_distribution_sorted.iloc[7:].sum()

# Creating a new DataFrame for visualization
pie_data = top_7[["region_id", "delivery_total"]].copy()
pie_data.loc[len(pie_data)] = ["Others", others["delivery_total"]]

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    pie_data["delivery_total"],
    labels=pie_data["region_id"],
    autopct="%1.1f%%",
    startangle=140,
    colors=plt.cm.Paired.colors
)
top_7 = region_delivery_distribution_sorted.head(7)[["region_id", "delivery_total"]]

# Displaying results in VS Code
print(top_7)

"""
Extra
"""

"""
test
"""
import pandas as pd
import matplotlib.pyplot as plt

# Extract the hour from delivery_time
df["delivery_hour"] = df["delivery_time"].dt.hour

# Count the number of deliveries per hour
hourly_distribution = df["delivery_hour"].value_counts().sort_index()

# Plot the bar chart
plt.figure(figsize=(12, 6))
plt.bar(hourly_distribution.index, hourly_distribution.values, color="midnightblue")

# Labels and title
plt.xlabel("Hour of the Day (0-24)")
plt.ylabel("Number of Deliveries")
plt.title("Distribution of Deliveries by Hour of the Day")
plt.xticks(range(0, 24))
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()

# Calculate total deliveries
total_deliveries = hourly_distribution.sum()

# Calculate deliveries after 18:00
deliveries_after_18 = hourly_distribution.loc[18:].sum()

# Calculate percentage of deliveries after 18:00
percentage_after_18 = (deliveries_after_18 / total_deliveries) * 100

# Display the result
percentage_after_18
