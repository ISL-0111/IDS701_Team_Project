** Reference **
<< top14 regions, log transformation, threshold 35 tasks >

<Threshold>


[Piecewise](https://github.com/ISL-0111/IDS701_Team_Project/blob/main/Data/regression_piecewise.py)

[Gam regression](https://github.com/ISL-0111/IDS701_Team_Project/blob/main/Data/regression_pygam.ipynb)

[ttest](https://github.com/ISL-0111/IDS701_Team_Project/blob/main/Experiment/ttest.ipynb)

[Regression](https://github.com/ISL-0111/IDS701_Team_Project/blob/main/Experiment/regresssion_task_count_threshold35_top14regions_log_0423.py)



**Research Question**

**Does assigning more tasks to a courier in a day increase the average delivery time per task, even after controlling for delivery distance and regional variation?**
https://docs.google.com/document/d/1FWvHJf7PiYHRKt898vxAXaCYm8M9A30BcwE5UStoIGs/edit?tab=t.46ivr93g3wxd#heading=h.upwtpbiusum

_Q. Does assigning more tasks to a courier in a day increase the average delivery time per task, even after controlling for delivery distance and regional variation?_

<Motivation> We initially explored the environmental implications of last-mile delivery operations, particularly in terms of CO₂ emissions. Based on those findings, we now focus on the operational factor most likely to influence emissions indirectly: delivery time. Our causal question investigates whether increasing task volume per courier leads to longer delivery times, which, in turn, could exacerbate emissions. This link allows us to move from exploratory observations to a policy-relevant causal mechanism/

Dataset: LastmileDelivery_Shanghai
Outcome Variable: Average Shipping Time taken per delivery task (delivery_time - accept_time) completed by a courier within a single day (courier_id + ds*)
* date
Treatment Variable: Number of packages assigned in a day (Need to be calculated)
Confounder: avg_distance_km (newly created*) Haversine Formula
* Computed as the direct distance between accept_gps and delivery_gps locations
Fixed Effects / Controls
region_id, aoi_id(area of interest?) : Geographic Fixed Effects
delivery hour zone: Select the most frequent hour during the day
Unit of Analysis: courier-day (courier_id + ds)
Estimation : covariate-adjusted linear regression (Using Continuous variable)
(e.g) avg_delivery_duration_minutes ~ task_count + avg_distance_km + C(region_id) + C(aoi_id) + C(delivery_hour_mode)

!! The values for region_id and aoi_id are just numeric codes with no public mapping to actual locations or district types, so they can only be used as fixed effects, not for interpretation. !!

* isolating the effect of increased task load on average delivery time per task, controlling for key confounding factors.

Implication(hypothetical) : Overloaded delivery workloads lead to longer task durations, which may increase CO₂ emissions through extended vehicle operation and idle time. This implies that task allocation optimization is not only relevant to operational efficiency, but also to environmental sustainability, particularly in the context of ESG-focused logistics operations.

**Memo**
- Using top 5 or 10 regionID (based on # of observations)
- Defining specific time span ? (1 week, 1 month etc) -> if so, it should be included in our control variable

(Step0) Data Preprocessing + EDA
(Step1) Identifying thresholds(# of deliveries a day per courierID),estimating the causal effect on delivery time using a continuous treatment model. Examining the marginal impact of assigning additional tasks
(Step2) Based on threshold from step1, Defining a binary treatment variable (overloaded vs. non-overloaded) and compare the outcomes using t-test

![image2](https://github.com/ISL-0111/IDS701_Team_Project/blob/main/Data/Others/Screenshot%202025-04-21%20at%209.58.28%E2%80%AFAM.png)
![image1](https://github.com/ISL-0111/IDS701_Team_Project/blob/main/Data/Others/Screenshot%202025-04-21%20at%209.58.48%E2%80%AFAM.png)


**Data**
LaDe: The First Comprehensive Last-mile Delivery Dataset from Industry
(Paper : https://arxiv.org/abs/2306.10675, Git : https://github.com/wenhaomin/LaDe) 
