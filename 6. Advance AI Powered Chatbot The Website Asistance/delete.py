import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime

# Define the timeline data
tasks = [
    ("Literature survey and dataset compilation", "2025-10", "2026-04"),
    ("Learn quantum chemistry tools and featurization", "2025-10", "2026-04"),
    ("Build initial property prediction models", "2025-12", "2026-04"),
    ("Preliminary benchmarking of ML models", "2026-05", "2026-07"),
    ("Pilot active learning loop on small dataset", "2026-05", "2026-07"),
    ("Expand dataset and retrain models", "2026-10", "2027-04"),
    ("Start explainability experiments", "2026-10", "2027-04"),
    ("First case study (organic dyes)", "2026-12", "2027-06"),
    ("Prepare conference paper on workflow/results", "2027-05", "2027-07"),
    ("Second case study (electrolyte materials)", "2027-10", "2028-04"),
    ("Pursue explainability deeply", "2027-10", "2028-04"),
    ("Begin writing thesis chapters", "2027-12", "2028-06"),
    ("Final integration of pipeline", "2028-05", "2028-07"),
    ("Draft full thesis", "2028-05", "2028-07"),
    ("Submit to journal(s)", "2028-06", "2028-08"),
    ("Complete final thesis write-up", "2028-10", "2029-01"),
    ("Submit final academic papers", "2028-10", "2029-01"),
    ("Oral defence and dissemination", "2028-10", "2029-01")
]

# Convert to DataFrame
df = pd.DataFrame(tasks, columns=["Task", "Start", "End"])
df["Start"] = pd.to_datetime(df["Start"])
df["End"] = pd.to_datetime(df["End"])
df["Duration"] = df["End"] - df["Start"]

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
for i, task in df.iterrows():
    ax.barh(i, task["Duration"].days, left=task["Start"], height=0.5, color="skyblue")
    ax.text(task["Start"], i, task["Task"], va='center', ha='left', fontsize=8)

# Format
ax.set_yticks(range(len(df)))
ax.set_yticklabels([""] * len(df))  # Hide y-axis labels
ax.invert_yaxis()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xlabel("Year")
plt.title("Gantt Chart – PhD Timeline")
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
