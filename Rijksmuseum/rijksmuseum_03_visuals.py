from matplotlib import gridspec
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import statsmodels.api as sm


# Account_Normalised_Growth: Line Graph with normalised Followers, Following and Number of Posts
# Posts_PostDistribution_PostType: Post Distribution by Post Type as a Pie Chart
# PM_Heatmap_Day_Hour: Heatmap with Day and Hour (excl. post_type = Reposts")
# PM_MeanEngagement_1k_PostType_Feature_Combination (excl. post_type = Reposts")
# PM_Avg_NormalisedEngagement_MediaType_EngagementType (excl. post_type = Reposts") Bar Chart
# PM_Avg_Engagement_1k_PostType (excl. post_type = Reposts") Bar Chart
# PM_Weekly_Engagement_GrowthRates_EngagementType (excl. post_type = Reposts") Line Graphs
# PM_Weekly_Engagement_GrowthRates_EngagementPostType (excl. post_type = Reposts") Line Graphs
# PM_Engagement_Composition_MediaType_PostType (excl. post_type = Reposts") Stacked Bar Chart
# PM_PostLength_Engagement_PostType (excl. post_type = Reposts") Scatter Plot
# PM_PostingFrequency_Engagement_PostType (excl. post_type = Reposts") Scatter Plot
# PM_ReplyType_PostDistribution: Reply Type Distribution as a Pie Chart
# PM_Avg_Engagement_ReplyType
# DH_WordCloud - Posts


# Set the default font size for all plots
mpl.rcParams["font.family"] = "Times New Roman"
AXIS_LABEL_SIZE = 11
TABLE_FONT_SIZE = 10
def format_thousands(x, _):
    return f"{int(x):,}".replace(",", " ")

# Set the default color palette for post types
color_dict = {
    "Original": "#87ceeb",  # Blue
    "Reply": "#799F79",     # Green
    "Quote": "#877B9E",     # Purple
    "Repost": "#B4B3B3",    # Grey
    "Blue": "#87ceeb",      # Blue
    "Green": "#799F79",     # Green
    "Purple": "#877B9E",    # Purple
    "Grey": "#B4B3B3",      # Grey
    "Brown": "#924F30",     # Brown
    "Orange": "#FF7538",   # Orange
    "Yellow": "#DED32B", # Yellow
    "Red": "#E3735E",       # Red
}


# Define data file paths
BASE_DIR = "/path/to/data"

account_data       = f"{BASE_DIR}/rijksmuseum_account.csv"
posts_data         = f"{BASE_DIR}/rijksmuseum_posts.csv"
posts_latest       = f"{BASE_DIR}/rijksmuseum_posts_latest_enriched_old.csv"
top10              = f"{BASE_DIR}/rijksmuseum_top10.csv"
worst10            = f"{BASE_DIR}/rijksmuseum_worst10.csv"
engagement_growth  = f"{BASE_DIR}/rijksmuseum_engagement_growth.csv"



# Account_Normalised_Growth: Line Graph with normalised Followers, Following and Number of Posts
# Load & process data
df = pd.read_csv(account_data, parse_dates=["execution_timestamp"])

df["follower_growth"] = (df["follower_count"] - df["follower_count"].iloc[0]) / df["follower_count"].iloc[0] * 100
df["following_growth"] = (df["following_count"] - df["following_count"].iloc[0]) / df["following_count"].iloc[0] * 100
df["posts_growth"] = (df["posts_count"] - df["posts_count"].iloc[0]) / df["posts_count"].iloc[0] * 100

# Plot setup
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))

sns.lineplot(data=df, x="execution_timestamp", y="follower_growth", label="Follower Growth", color="#87ceeb")
sns.lineplot(data=df, x="execution_timestamp", y="following_growth", label="Following Growth", color="#799F79")
sns.lineplot(data=df, x="execution_timestamp", y="posts_growth", label="Posts Growth", color="#877B9E")

ax.set_title("Rijksmuseum: Normalised Growth of Followers, Following & Posts", fontsize=13, fontweight="bold")
ax.set_xlabel("Date", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
ax.set_ylabel("Growth (%)", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=30, ha='right')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.set_ylim(0, None)
ax.legend()

# Table data
table_data = [
    [
        "Followers",
        f"{df['follower_count'].iloc[0]:,}",
        f"{df['follower_count'].iloc[-1]:,}",
        f"{df['follower_count'].iloc[-1] - df['follower_count'].iloc[0]:,}",
        f"{((df['follower_count'].iloc[-1] - df['follower_count'].iloc[0]) / df['follower_count'].iloc[0] * 100):.2f}%",
    ],
    [
        "Following",
        f"{df['following_count'].iloc[0]:,}",
        f"{df['following_count'].iloc[-1]:,}",
        f"{df['following_count'].iloc[-1] - df['following_count'].iloc[0]:,}",
        f"{((df['following_count'].iloc[-1] - df['following_count'].iloc[0]) / df['following_count'].iloc[0] * 100):.2f}%",
    ],
    [
        "Posts",
        f"{df['posts_count'].iloc[0]:,}",
        f"{df['posts_count'].iloc[-1]:,}",
        f"{df['posts_count'].iloc[-1] - df['posts_count'].iloc[0]:,}",
        f"{((df['posts_count'].iloc[-1] - df['posts_count'].iloc[0]) / df['posts_count'].iloc[0] * 100):.2f}%",
    ],
]

# Table 
table = plt.table(
    cellText=table_data,
    colLabels=["Metric", df['execution_timestamp'].iloc[0].date(), df['execution_timestamp'].iloc[-1].date(), "Growth in Numbers", "Growth in %"],
    cellLoc="center",
    loc="bottom",
    bbox=[0, -0.55, 1, 0.3],
    colColours=["lightgrey"] * 5
)

table.auto_set_font_size(False)
table.set_fontsize(TABLE_FONT_SIZE)
table.scale(1.2, 1.2)

# Final layout adjustments 
plt.tight_layout()
plt.subplots_adjust(bottom=0.5)  # ensures space between x-axis and table
plt.show()


# Posts_PostDistribution_PostType: Post Distribution by Post Type as a Pie Chart
# Load data
df = pd.read_csv(posts_latest)

# Count post types
counts = df["post_type"].value_counts()
percentages = (counts / counts.sum() * 100).round(2)
labels = counts.index.tolist()
colours = [color_dict.get(pt, "#cccccc") for pt in labels]

# Plot
fig, ax = plt.subplots(figsize=(8, 8))

wedges, texts, autotexts = ax.pie(
    counts,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    colors=colours
)

# Style labels
for t in texts:
    t.set_fontsize(AXIS_LABEL_SIZE)
    t.set_fontweight("bold")
for at in autotexts:
    at.set_fontsize(AXIS_LABEL_SIZE - 1)

# Title
plt.title("Rijksmuseum: Post Distribution by Post Type", fontsize=13, fontweight="bold")

# Table below
table_data = [
    [label, f"{counts[label]}", f"{percentages[label]:.2f}%"]
    for label in labels
]

table = plt.table(
    cellText=table_data,
    colLabels=["Post Type", "Count", "%"],
    loc="bottom",
    cellLoc="center",
    colColours=["lightgrey"] * 3,
    bbox=[0, -0.4, 1, 0.25]
)
table.auto_set_font_size(False)
table.set_fontsize(TABLE_FONT_SIZE)
table.scale(1.2, 1.2)

# Adjust spacing
plt.tight_layout()
plt.subplots_adjust(bottom=0.35)
plt.show()


# PM_Heatmap_Day_Hour: Heatmap with Day and Hour (excl. post_type = Reposts")
# Load dataset
df = pd.read_csv(posts_latest)

# Convert 'created_at' to datetime
df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', utc=True)

# Filter out Reposts
df = df[df['post_type'] != 'Repost']

# Extract day name and hour
df['day'] = df['created_at'].dt.day_name()
df['hour'] = df['created_at'].dt.hour

# Define fixed order for days and hours
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
hour_order = list(range(0, 24))
df['day'] = pd.Categorical(df['day'], categories=day_order, ordered=True)
df['hour'] = pd.Categorical(df['hour'], categories=hour_order, ordered=True)

# Define post types and blue colour map
post_types = ['Original', 'Reply']
cmap = sns.light_palette("dodgerblue", as_cmap=True)

# HEATMAP PLOTS
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), sharey=True)

for i, ptype in enumerate(post_types):
    subset = df[df['post_type'] == ptype]

    if subset.empty:
        axes[i].axis("off")
        axes[i].set_title(f"No data for {ptype}", fontsize=12, fontweight='bold')
        continue

    pivot = (
        subset.groupby(['day', 'hour'], observed=True)['engagement_score']
        .mean()
        .unstack(fill_value=0)
        .reindex(index=day_order)
    )

    if pivot.empty:
        axes[i].axis("off")
        axes[i].set_title(f"No data for {ptype}", fontsize=12, fontweight='bold')
        continue

    sns.heatmap(
        pivot,
        ax=axes[i],
        cmap=cmap,
        linewidths=0.4,
        linecolor='white',
        cbar=i == 2,
        square=False
    )

    axes[i].set_title(f"{ptype} Posts", fontsize=12, fontweight='bold')
    axes[i].set_xlabel("Hour of Day", fontweight='bold')
    if i == 0:
        axes[i].set_ylabel("Day of Week", fontweight='bold')

fig.suptitle("Rijksmuseum: Avg. Engagement of Posts by Day & Hour Posted", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# PM_MeanEngagement_1k_PostType_Feature_Combination (excl. post_type = Reposts")
# Load data
posts = pd.read_csv(posts_latest)
account = pd.read_csv(account_data, parse_dates=["execution_timestamp"])

# Normalisation factor (followers / 1 000)
def latest_followers(path: str) -> float | None:
    if not os.path.isfile(path):
        return None
    acc = pd.read_csv(path, parse_dates=["execution_timestamp"])
    latest = acc.sort_values("execution_timestamp").iloc[-1]
    return latest.get("follower_count", np.nan)

followers = latest_followers(account_data)
norm_divisor = followers / 1_000 if followers and not np.isnan(followers) else 1
if norm_divisor == 1:
    print("follower_count unavailable → raw engagement shown")

# Data prep
df = posts[posts["post_type"] != "Repost"].copy()

df["has_mentions"] = df["mention_count"].gt(0)
df["has_url"]      = df["urls"].notna() & df["urls"].ne("")
df["has_media"]    = df["media"].notna() & df["media"].ne("[]")

def classify(row):
    features = []
    if row["has_mentions"]:
        features.append("Mentions")
    if row["has_url"]:
        features.append("Link")  # Changed "URL" to "Link"
    if row["has_media"]:
        features.append("Media")
    return " + ".join(features) if features else "No Features"

df["engagement_group"] = df.apply(classify, axis=1)

df["total_engagement"] = df[["likes_count", "replies_count", "reposts_count"]].sum(axis=1)
df["eng_per_1k_followers"] = df["total_engagement"] / norm_divisor

# Aggregation by post_type and engagement_group (all combinations)
agg = (
    df.groupby(["post_type", "engagement_group"])["eng_per_1k_followers"]
    .mean()
    .reset_index(name="avg_eng_per_1k")
)

# Logical order of feature combinations for consistent legend and table columns
group_order = [
    "No Features", "Mentions", "Link", "Media",
    "Mentions + Link", "Mentions + Media", "Link + Media",
    "Mentions + Link + Media"
]
present_groups = [g for g in group_order if g in agg["engagement_group"].unique()]

# Define engagement group colour mapping using custom palette
engagement_group_colours = {
    "No Features": color_dict["Blue"],
    "Mentions": color_dict["Grey"],
    "Link": color_dict["Purple"],
    "Media": color_dict["Green"],
    "Mentions + Link": color_dict["Orange"],
    "Mentions + Media": color_dict["Red"],
    "Link + Media": color_dict["Yellow"],
    "Mentions + Link + Media": color_dict["Brown"],
}

# Get colours for only the groups present in the data
palette = [engagement_group_colours[g] for g in present_groups]


# Plot: Grouped bar chart + numeric table
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

# Barplot
ax = fig.add_subplot(gs[0])
sns.barplot(
    data=agg,
    x="post_type",
    y="avg_eng_per_1k",
    hue="engagement_group",
    hue_order=present_groups,
    palette=palette,
    ax=ax
)

ax.set_title(f"Rijksmuseum: Mean Engagement per 1k Followers\nby Post Type & Feature Combination", fontsize=14, fontweight="bold", pad=12, y=1.02)
ax.set_xlabel("Post Type", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
ax.set_ylabel("Avg Engagement / 1k Followers", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
ax.tick_params(axis='x', rotation=15)
ax.legend(title="Feature Combination", bbox_to_anchor=(1.05, 1), loc='upper left')

# Calculate counts per group for annotation in the table
counts = (
    df.groupby(["post_type", "engagement_group"])
    .size()
    .reset_index(name="post_count")
)

# Merge mean engagement with counts
agg_counts = pd.merge(
    agg,
    counts,
    on=["post_type", "engagement_group"],
    how="left"
)

# Prepare pivot table for the table below plot
pivot_table = (
    agg_counts.assign(
        display=lambda d: d["avg_eng_per_1k"].round(2).astype(str) + 
                         " (" + d["post_count"].astype(str) + ")"
    )
    .pivot(index="post_type", columns="engagement_group", values="display")
    .reindex(columns=present_groups)
    .reset_index()
)

# Convert to list of lists for plt.table
table_data = pivot_table.values.tolist()

# Column labels with light grey background
col_labels = ["Post Type"] + present_groups
col_colours = ["lightgrey"] * len(col_labels)

# Add table below the bar chart
table = plt.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="bottom",
    bbox=[0, -0.5, 1, 0.3],
    colColours=col_colours,
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.2)

# Center text and bold headers
for (row, col), cell in table.get_celld().items():
    cell.set_text_props(ha="center", va="center")
    if row == 0:
        cell.set_text_props()

plt.tight_layout()
plt.show()

# PM_Avg_NormalisedEngagement_MediaType_EngagementType (excl. post_type = Reposts") Bar Chart
# Load data
posts = pd.read_csv(posts_latest)
account = pd.read_csv(account_data, parse_dates=["execution_timestamp"])

# Filter out reposts
df = posts[posts['post_type'] != 'Repost'].copy()

# Clean media type
df['media_type'] = df['media'].fillna('Text-only')
df.loc[df['media_type'].isin(['', '[]']), 'media_type'] = 'Text-only'

# Total engagement per post
df['total_engagement'] = df[['likes_count', 'replies_count', 'reposts_count']].sum(axis=1)

# Get latest follower count for normalisation
latest_account = account.sort_values("execution_timestamp").iloc[-1]
followers = latest_account.get("follower_count", np.nan)
norm_divisor = followers / 1_000 if followers and not np.isnan(followers) else 1
if norm_divisor == 1:
    print("follower_count unavailable → using raw engagement")

# Normalised engagement
df['engagement_norm'] = df['total_engagement'] / norm_divisor

# Aggregate engagement by media type and type
agg_split = df.groupby('media_type').agg(
    avg_likes_norm=('likes_count', lambda x: x.mean() / norm_divisor),
    avg_replies_norm=('replies_count', lambda x: x.mean() / norm_divisor),
    avg_reposts_norm=('reposts_count', lambda x: x.mean() / norm_divisor),
    count=('media_type', 'size')
).reset_index()

# Total average engagement
agg_split['avg_total_engagement'] = (
    agg_split['avg_likes_norm'] + agg_split['avg_replies_norm'] + agg_split['avg_reposts_norm']
)
agg_split['percent'] = agg_split['count'] / agg_split['count'].sum() * 100

# Sort ascending by total average engagement (least at top)
agg_split = agg_split.sort_values('avg_total_engagement', ascending=True).reset_index(drop=True)

# Prepare data for seaborn (melt)
melted = agg_split.melt(
    id_vars='media_type',
    value_vars=['avg_replies_norm', 'avg_reposts_norm', 'avg_likes_norm'],  # order matches desired stack order
    var_name='engagement_type',
    value_name='average_engagement'
)

# Palette matching engagement types
palette = {
    'avg_replies_norm': '#799F79',   # green
    'avg_reposts_norm': '#B4B3B3',   # grey
    'avg_likes_norm': '#87ceeb'      # blue
}

# Plot setup
fig = plt.figure(figsize=(12, 9))
gs = GridSpec(2, 1, height_ratios=[3, 1])

# Bar plot
ax_bar = fig.add_subplot(gs[0])
sns.barplot(
    data=melted,
    y='media_type',
    x='average_engagement',
    hue='engagement_type',
    palette=palette,
    ax=ax_bar,
    order=agg_split['media_type'],
    hue_order=['avg_replies_norm', 'avg_reposts_norm', 'avg_likes_norm']
)

ax_bar.set_title(f"Rijksmuseum: Average Normalised Engagement by Media Type and Engagement Type", fontsize=14, fontweight="bold", pad=12)
ax_bar.set_xlabel("Average Engagement per 1k Followers", fontsize=12, fontweight="bold")
ax_bar.set_ylabel("Media Type", fontsize=12, fontweight="bold")

# Custom legend matching hue order and colors
legend_handles = [
    Patch(color=palette['avg_replies_norm'], label="Replies"),
    Patch(color=palette['avg_reposts_norm'], label="Reposts"),
    Patch(color=palette['avg_likes_norm'], label="Likes"),
]
ax_bar.legend(handles=legend_handles, title="Engagement Type", loc="upper right")

# Table with rounded values
table_data = agg_split.copy()
round_cols = ['avg_likes_norm', 'avg_replies_norm', 'avg_reposts_norm', 'avg_total_engagement', 'percent']
table_data[round_cols] = table_data[round_cols].round(2)

table_values = table_data[
    ['media_type', 'avg_total_engagement', 'avg_likes_norm', 'avg_replies_norm', 'avg_reposts_norm', 'count', 'percent']
].values.tolist()

col_labels = ["Media Type", "Avg Total Engagement", "Avg Likes", "Avg Replies", "Avg Reposts", "Count", "% of Posts"]

ax_table = fig.add_subplot(gs[1])
ax_table.axis("off")
table = ax_table.table(
    cellText=table_values,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    colColours=["lightgrey"] * len(col_labels)
)
table.auto_set_font_size(False)
table.set_fontsize(TABLE_FONT_SIZE)
table.scale(1, 1.4)

for (row, col), cell in table._cells.items():
    cell.set_text_props(ha='center', va='center')
    if row == 0:
        cell.set_text_props()

plt.tight_layout()
plt.show()


# PM_Avg_Engagement_1k_PostType
# Load data
posts = pd.read_csv(posts_latest)
account = pd.read_csv(account_data, parse_dates=["execution_timestamp"])

# Get normalisation divisor
def latest_followers(path: str) -> float | None:
    if not os.path.isfile(path):
        return None
    acc = pd.read_csv(path, parse_dates=["execution_timestamp"])
    latest = acc.sort_values("execution_timestamp").iloc[-1]
    return latest.get("follower_count", np.nan)

followers = latest_followers(account_data)
norm_divisor = followers / 1_000 if followers and not np.isnan(followers) else 1
if norm_divisor == 1:
    print("follower_count unavailable → raw engagement shown")

# Filter and normalise
df = posts[posts["post_type"] != "Repost"].copy()

df["likes_norm"]    = df["likes_count"]    / norm_divisor
df["replies_norm"]  = df["replies_count"]  / norm_divisor
df["reposts_norm"]  = df["reposts_count"]  / norm_divisor

# Aggregation by post_type
agg = (
    df.groupby("post_type")[["likes_norm", "replies_norm", "reposts_norm"]]
    .mean()
    .reset_index()
    .melt(id_vars="post_type", var_name="engagement_type", value_name="mean_per_1k")
)

# Map nicer labels
engagement_label_map = {
    "likes_norm": "Likes",
    "replies_norm": "Replies",
    "reposts_norm": "Reposts"
}
agg["engagement_type"] = agg["engagement_type"].map(engagement_label_map)

# Custom colour palette
custom_palette = {
    "Replies": '#799F79',   # green
    "Reposts": '#B4B3B3',   # grey
    "Likes":   '#87ceeb'    # blue
}

# Plot setup (use set_theme instead of deprecated set)
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(12, 9))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

# Barplot
ax = fig.add_subplot(gs[0])
sns.barplot(
    data=agg,
    x="post_type",
    y="mean_per_1k",
    hue="engagement_type",
    palette=custom_palette,
    ax=ax
)

# Prettier axis labels (does not change column names!)
ax.set_xlabel("Post Type", fontsize=12, fontweight="bold")
ax.set_ylabel("Mean Engagement / 1k Followers", fontsize=12, fontweight="bold")

ax.set_title(f"Rijksmuseum: Average Engagement per 1k Followers by Post Type", fontsize=14, fontweight="bold", pad=12)

# Legend title
ax.legend(title="Engagement Type", title_fontsize=11, fontsize=10)

# Prepare table data
pivot_table = (
    agg.pivot(index="post_type", columns="engagement_type", values="mean_per_1k")
    .round(2)
    .reset_index()
)
col_labels = ["Post Type"] + list(pivot_table.columns[1:])
col_colours = ["lightgrey"] * len(col_labels)
table_data = pivot_table.values.tolist()

# Add table
table = plt.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="bottom",
    bbox=[0, -0.5, 1, 0.3],
    colColours=col_colours,
)

table.auto_set_font_size(False)
table.fontsize = TABLE_FONT_SIZE
table.scale(1.2, 1.2)

# Header
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(ha="center", va="center")

plt.tight_layout()
plt.show()


# PM_Weekly_Engagement_GrowthRates_EngagementType (excl. post_type = Reposts")
# Load data
df_growth = pd.read_csv(engagement_growth)
df_latest = pd.read_csv(posts_latest)

# Clean datetime
df_growth["first_timestamp"] = pd.to_datetime(df_growth["first_timestamp"], errors="coerce")
df_growth["last_timestamp"] = pd.to_datetime(df_growth["last_timestamp"], errors="coerce")
df_growth = df_growth.dropna(subset=["first_timestamp", "last_timestamp"])

# Remove reposts
repost_uris = df_latest.loc[df_latest["post_type"] == "Repost", "uri"].unique()
df_growth = df_growth.loc[~df_growth["uri"].isin(repost_uris)]

# Remove all-zero rows
growth_cols = [
    "likes_growth_pct",
    "reposts_growth_pct",
    "replies_growth_pct",
    "engagement_score_growth_pct"
]
df_growth = df_growth.loc[~(df_growth[growth_cols] == 0).all(axis=1)]

# Weekly aggregation based on execution (last) timestamp
df_growth["week"] = df_growth["last_timestamp"].dt.to_period("W").dt.start_time
weekly_avg = (
    df_growth.groupby("week")[growth_cols]
    .mean()
    .reset_index()
    .melt(id_vars="week", var_name="metric", value_name="avg_growth")
)

# Map to readable labels
label_map = {
    "likes_growth_pct": "Likes",
    "reposts_growth_pct": "Reposts",
    "replies_growth_pct": "Replies",
    "engagement_score_growth_pct": "Overall Engagement"
}
weekly_avg["metric"] = weekly_avg["metric"].map(label_map)

# Colour palette
palette = {
    "Likes": "#87CEEB",
    "Reposts": "#B4B3B3",
    "Replies": "#799F79",
    "Overall Engagement": "#924F30"
}

# Prepare enhanced summary table
summary_stats = weekly_avg.groupby("metric")["avg_growth"].agg(
    ["mean", "median", "min", "max"]
).round(2).reset_index()

# Rename columns for better readability
summary_stats.columns = ["Metric", "Mean", "Median", "Min", "Max"]

# Pivot the table to get the desired format
pivot_table = summary_stats.set_index("Metric")
pivot_table = pivot_table.reindex(["Likes", "Reposts", "Replies", "Overall Engagement"])

# Convert to list of lists for plt.table
table_data = pivot_table.reset_index().values.tolist()

# Column labels with light grey background
col_labels = ["Metric", "Mean", "Median", "Min", "Max"]
col_colours = ["lightgrey"] * len(col_labels)

# Create the figure
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 1, height_ratios=[3, 1])
ax = fig.add_subplot(gs[0])

# Line plot
sns.lineplot(
    data=weekly_avg,
    x="week",
    y="avg_growth",
    hue="metric",
    palette=palette,
    marker="o",
    ax=ax
)

# Grid and tick formatting
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.5)

# Add table below the plot
table = plt.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="bottom",
    bbox=[0, -0.5, 1, 0.25],
    colColours=col_colours,
)

table.auto_set_font_size(False)
table.set_fontsize(TABLE_FONT_SIZE)
table.scale(1.2, 1.2)

# Center text and bold headers
for (row, col), cell in table.get_celld().items():
    cell.set_text_props(ha="center", va="center")
    if row == 0:
        cell.set_text_props(fontproperties=mpl.font_manager.FontProperties())

# Labels
ax.set_title("Rijksmuseum: Weekly Engagement Growth Rates by Type", fontsize=14, fontweight="bold", pad=16)
ax.set_xlabel("Week", fontsize=12, fontweight="bold", labelpad=12)
ax.set_ylabel("Average Growth Rate (%)", fontsize=12, fontweight="bold")
ax.legend(title="Engagement Type", title_fontsize=11, fontsize=10)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# PM_Weekly_Engagement_GrowthRates_EngagementPostType (excl. post_type = Reposts")
# Load data
df_growth = pd.read_csv(engagement_growth)
df_latest = pd.read_csv(posts_latest)

# Clean datetime
df_growth["first_timestamp"] = pd.to_datetime(df_growth["first_timestamp"], errors="coerce")
df_growth["last_timestamp"] = pd.to_datetime(df_growth["last_timestamp"], errors="coerce")
df_growth = df_growth.dropna(subset=["first_timestamp", "last_timestamp"])

# Remove reposts
repost_uris = df_latest.loc[df_latest["post_type"] == "Repost", "uri"].unique()
df_growth = df_growth.loc[~df_growth["uri"].isin(repost_uris)]

# Add post type information to df_growth
df_growth = df_growth.merge(df_latest[["uri", "post_type"]], on="uri", how="left")

# Remove all-zero rows
growth_cols = [
    "likes_growth_pct",
    "reposts_growth_pct",
    "replies_growth_pct",
    "engagement_score_growth_pct"
]
df_growth = df_growth.loc[~(df_growth[growth_cols] == 0).all(axis=1)]

# Weekly aggregation based on execution (last) timestamp
df_growth["week"] = df_growth["last_timestamp"].dt.to_period("W").dt.start_time
weekly_avg = (
    df_growth.groupby(["week", "post_type"])[growth_cols]
    .mean()
    .reset_index()
    .melt(id_vars=["week", "post_type"], var_name="metric", value_name="avg_growth")
)

# Map to readable labels
label_map = {
    "likes_growth_pct": "Likes",
    "reposts_growth_pct": "Reposts",
    "replies_growth_pct": "Replies",
    "engagement_score_growth_pct": "Overall Engagement"
}
weekly_avg["metric"] = weekly_avg["metric"].map(label_map)

# Colour palette
palette = {
    "Likes": "#87CEEB",
    "Reposts": "#B4B3B3",
    "Replies": "#799F79",
    "Overall Engagement": "#924F30"
}

# Get unique post types
post_types = weekly_avg["post_type"].unique()

# Define custom order for post types
post_types_order = ["Original", "Reply"]

# Create a figure with subplots for each post type
fig, axes = plt.subplots(nrows=1, ncols=len(post_types_order), figsize=(18, 6), sharey=True)

# Plot for each post type
for ax, post_type in zip(axes, post_types_order):
    subset = weekly_avg[weekly_avg["post_type"] == post_type]
    sns.lineplot(
        data=subset,
        x="week",
        y="avg_growth",
        hue="metric",
        palette=palette,
        marker="o",
        ax=ax
    )
    ax.set_title(f"Rijksmuseum: {post_type}", fontweight="bold")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.5)
    ax.set_xlabel("Week", fontweight="bold", fontsize=AXIS_LABEL_SIZE, labelpad=12)
    ax.set_ylabel("Average Growth Rate (%)", fontweight="bold", fontsize=AXIS_LABEL_SIZE, labelpad=12)
    ax.xaxis.set_label_coords(0.5, -0.15)
    ax.legend(title="Engagement Type", loc="lower right")

    # Rotate x-axis labels only for 'Original'
    if post_type == "Original":
        for label in ax.get_xticklabels():
            label.set_rotation(45)

 
# Summary table with mean and max values
summary_mean = weekly_avg.groupby(["post_type", "metric"])["avg_growth"].mean().unstack()
summary_max = weekly_avg.groupby(["post_type", "metric"])["avg_growth"].max().unstack()

# Combine mean and max into a single DataFrame
summary = pd.DataFrame({
    "Mean Likes": summary_mean["Likes"],
    "Max Likes": summary_max["Likes"],
    "Mean Reposts": summary_mean["Reposts"],
    "Max Reposts": summary_max["Reposts"],
    "Mean Replies": summary_mean["Replies"],
    "Max Replies": summary_max["Replies"],
    "Mean Overall Engagement": summary_mean["Overall Engagement"],
    "Max Overall Engagement": summary_max["Overall Engagement"]
}).round(2)

# Table data
table_data = summary.reset_index().values.tolist()
col_labels = ["Post Type", "Mean Likes", "Max Likes", "Mean Reposts", "Max Reposts", "Mean Replies", "Max Replies", "Mean Overall Engagement", "Max Overall Engagement"]

# Add central table below plots
table_ax = fig.add_axes([0.1, 0.02, 0.8, 0.15])
table_ax.axis("off")
table = table_ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    colColours=["lightgrey"] * len(col_labels)
)

table.auto_set_font_size(False)
table.set_fontsize(TABLE_FONT_SIZE)
table.scale(1.2, 1.2)

fig.suptitle("Rijksmuseum: Weekly Engagement Growth Rates by Engagement and Post Type", fontsize=14, fontweight="bold", y=.97)


plt.subplots_adjust(bottom=0.25, top=0.9)
plt.show()

# PM_Engagement_Composition_MediaType_PostType
# Load latest posts
df_latest = pd.read_csv(posts_latest)

# Filter out reposts and rows without media
df_latest = df_latest[df_latest["post_type"] != "Repost"]
df_latest = df_latest.dropna(subset=["media"])

# Clean strings
df_latest["media"] = df_latest["media"].str.strip().str.lower()
df_latest["post_type"] = df_latest["post_type"].str.strip()

# Group by post_type and media
grouped = df_latest.groupby(["post_type", "media"])[
    ["likes_count", "reposts_count", "replies_count"]
].sum()

# Rename columns
grouped = grouped.rename(columns={
    "likes_count": "Likes",
    "reposts_count": "Reposts",
    "replies_count": "Replies",
})

# Convert to percentages per group (sum across Likes, Reposts, Replies)
total_per_group = grouped.sum(axis=1)
grouped_pct = grouped.div(total_per_group, axis=0) * 100

# Engagement types & colors
engagement_types = ["Likes", "Reposts", "Replies"]
colors = {
    "Likes": "#87CEEB",
    "Reposts": "#B4B3B3",
    "Replies": "#799F79"
}

post_types = grouped_pct.index.get_level_values(0).unique()
n_post_types = len(post_types)

# Create figure with GridSpec: 2 rows, n_post_types columns (top), 1 row spanning all columns (bottom)
fig = plt.figure(figsize=(6 * n_post_types, 9))
gs = GridSpec(3, n_post_types, height_ratios=[4, 0.1, 1], figure=fig)

# Bar plot axes on row 0
axes = [fig.add_subplot(gs[0, i]) for i in range(n_post_types)]

# Legend axis on row 1 spanning all columns
legend_ax = fig.add_subplot(gs[1, :])
legend_ax.axis('off')  # hide axis

# Table axis on row 2 spanning all columns
table_ax = fig.add_subplot(gs[2, :])
table_ax.axis('off')

# Plot stacked bars
for ax, post_type in zip(axes, post_types):
    subset = grouped_pct.loc[post_type]
    bottom = pd.Series(0, index=subset.index)
    for metric in engagement_types:
        bars = ax.bar(
            x=subset.index,
            height=subset[metric],
            bottom=bottom,
            label=metric,
            color=colors[metric]
        )
        for bar, pct in zip(bars, subset[metric]):
            if pct > 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{pct:.0f}%",
                    ha='center',
                    va='center',
                    fontsize=9,
                )
        bottom += subset[metric]

    ax.set_title("Rijksmuseum: post_type", fontsize=12, fontweight="bold")
    ax.set_xlabel("Media Type", fontweight="bold")
    ax.set_xticks(range(len(subset.index)))
    ax.set_xticklabels(subset.index, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if ax == axes[0]:
        ax.set_ylabel("Share of Engagement (%)", fontweight="bold")
        handles, labels = ax.get_legend_handles_labels()

fig.suptitle("Rijksmuseum: Engagement Composition by Media Type and Post Type", fontsize=14, fontweight="bold", y=0.95)

# Add legend in dedicated legend_ax
legend = legend_ax.legend(
    handles, labels,
    title="Engagement Type",
    loc="center",
    ncol=len(engagement_types),
    frameon=False
)

# Prepare summary table data (Likes, Reposts, Replies)
summary_df = grouped_pct.reset_index()

# Pivot wider for each engagement type, concatenate columns with multi-level columns
likes = summary_df.pivot(index="post_type", columns="media", values="Likes")
reposts = summary_df.pivot(index="post_type", columns="media", values="Reposts")
replies = summary_df.pivot(index="post_type", columns="media", values="Replies")

# Create a multi-index dataframe for the table: columns = MultiIndex (metric, media)
table_data = pd.concat(
    [likes, reposts, replies],
    keys=engagement_types,
    axis=1
)

# Swap column MultiIndex levels to have media first, then metric
table_data_swapped = table_data.copy()
table_data_swapped.columns = table_data_swapped.columns.swaplevel(0, 1)
table_data_swapped = table_data_swapped.sort_index(axis=1, level=[0,1])

# Flatten columns to strings like "External Thumbnails\nLikes" for line break
table_data_swapped.columns = [
    f"{media.title().replace('_', ' ')}\n{metric}"
    for media, metric in table_data_swapped.columns
]
# Format as percentages with no decimals
table_display = table_data_swapped.stack().map(
    lambda x: f"{x:.0f}%" if pd.notnull(x) else ""
).unstack()

# Prepare data for table: add Post Type as first column in the cellText matrix
post_types_list = table_display.index.tolist()
cell_text = [[post_type] + list(row) for post_type, row in zip(post_types_list, table_display.values)]

# Prepare column labels: add "Post Type" header first, then existing column labels
col_labels = ["Post Type"] + list(table_display.columns)

# Draw table on table_ax
tbl = table_ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc='center',
    rowLoc='center',
    loc='center',
    colColours=["#f0f0f0"] * len(col_labels)
)

# Increase header row height
header_row = 0
header_height = 0.35  
for col in range(len(col_labels)):
    tbl[(header_row, col)].set_height(header_height)

tbl.auto_set_font_size(False)
tbl.set_fontsize(TABLE_FONT_SIZE)
tbl.scale(1, 1.5)

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.show()

# PM_PostLength_Engagement_PostType
# Load data
df_posts = pd.read_csv(posts_latest)
df_account = pd.read_csv(account_data)

# Get latest follower count
latest_follower_count = df_account.sort_values("execution_timestamp").iloc[-1]["follower_count"]

# Filter relevant post types
df_posts = df_posts[df_posts["post_type"].isin(["Original", "Reply"])]

# Character count
df_posts["char_count"] = df_posts["text"].fillna("").apply(len)

# Engagement per 1k followers
for metric in ["likes_count", "reposts_count", "replies_count"]:
    df_posts[f"{metric}_per_1k"] = df_posts[metric] / (latest_follower_count / 1000)

# Setup: Post types as rows, metrics as columns
post_types = ["Original", "Reply"]
metrics = [
    ("likes_count_per_1k", "Likes per 1k Followers"),
    ("reposts_count_per_1k", "Reposts per 1k Followers"),
    ("replies_count_per_1k", "Replies per 1k Followers")
]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 14), sharex=True)

# Plot
for i, post_type in enumerate(post_types):
    subset = df_posts[df_posts["post_type"] == post_type]
    for j, (metric, _) in enumerate(metrics):
        ax = axes[i, j]
        sns.regplot(
            data=subset,
            x="char_count",
            y=metric,
            lowess=True,
            scatter_kws={"alpha": 0.5, "s": 20},
            line_kws={"color": "black"},
            ax=ax
        )
        ax.grid(True, linestyle="--", alpha=0.6)

        # Axis formatting
        if i < 2:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Character Count", fontsize=11, fontweight="bold")
        if j > 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel(post_type, fontsize=11, fontweight="bold")

        ax.set_title("")

# Set column headers
for ax, (_, col_title) in zip(axes[0], metrics):
    ax.set_title(col_title, fontsize=12, fontweight="bold", pad=15)

# Compute summary stats including std
summary = df_posts.groupby("post_type")[
    ["likes_count_per_1k", "reposts_count_per_1k", "replies_count_per_1k"]
].agg(["mean", "median", "std"]).round(2)

# Flatten MultiIndex and rename column
summary.columns = [
    "Likes (mean)", "Likes (median)", "Likes (std)",
    "Reposts (mean)", "Reposts (median)", "Reposts (std)",
    "Replies (mean)", "Replies (median)", "Replies (std)"
]
summary = summary.reset_index()
summary.rename(columns={"post_type": "Post Type"}, inplace=True)

# Adjust layout manually
plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.3)

# Add table below the plot
table_ax = fig.add_axes([0.1, 0.02, 0.8, 0.15])  # [left, bottom, width, height]
table_ax.axis("off")

tbl = table_ax.table(
    cellText=summary.values,
    colLabels=summary.columns,
    cellLoc="center",
    loc="center",
    colColours=["lightgrey"] * len(summary.columns)
)

# Uniformly increase vertical scaling first (optional base)
tbl.scale(1.1, 2.0)  # x scale, y scale (increase y scale more for taller rows)

# Then manually adjust row heights
# Get all cells: keys are (row, col)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        # Header row: set bigger height
        cell.set_height(0.2)  # Increase header height
    else:
        # Other rows: a bit smaller height but still bigger than default
        cell.set_height(0.1)

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.1, 1.5)

# Main title
plt.suptitle("Rijksmuseum: Post Length vs Engagement (per 1k Followers) by Post Type", fontsize=15, fontweight="bold")
plt.show()

# PM_PostingFrequency_Engagement_PostType
# Load data
df_posts = pd.read_csv(posts_latest, parse_dates=["created_at"], dayfirst=False)
df_account = pd.read_csv(account_data)

# Clean and prepare data
df_posts["created_at"] = pd.to_datetime(df_posts["created_at"], errors="coerce")
df_posts = df_posts.dropna(subset=["created_at"])
df_posts["created_at_naive"] = df_posts["created_at"].dt.tz_localize(None)
df_posts["week"] = df_posts["created_at_naive"].dt.to_period("W").dt.start_time

df_account["execution_timestamp"] = pd.to_datetime(df_account["execution_timestamp"], errors="coerce")
latest_follower_count = df_account.sort_values("execution_timestamp").iloc[-1]["follower_count"]

# Filter out reposts
df_filtered = df_posts[df_posts["post_type"] != "Repost"].copy()

# Calculate engagement
df_filtered["engagement_score"] = (
    df_filtered["likes_count"] + df_filtered["reposts_count"] + df_filtered["replies_count"]
)
df_filtered["engagement_per_1k"] = df_filtered["engagement_score"] / (latest_follower_count / 1000)

# Define post types
post_types = ["Original", "Reply"]

# Create figure and GridSpec
fig = plt.figure(figsize=(18, 8))
gs = GridSpec(2, 2, height_ratios=[6, 1], hspace=0.4, figure=fig)

axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
table_ax = fig.add_subplot(gs[1, :])
table_ax.axis("off")

for ax, ptype in zip(axes, post_types):
    df_ptype = df_filtered[df_filtered["post_type"] == ptype]
    weekly_summary = df_ptype.groupby("week").agg(
        post_count=("uri", "count"),
        avg_engagement_per_1k=("engagement_per_1k", "mean")
    ).reset_index()

    # Scatter plot
    ax.scatter(
        weekly_summary["post_count"],
        weekly_summary["avg_engagement_per_1k"],
        alpha=0.7,
        s=60,
        edgecolor='k'
    )

    # LOWESS smoothing – safe application
    filtered = weekly_summary.dropna(subset=["post_count", "avg_engagement_per_1k"])
    if (
        len(filtered) >= 5 and
        filtered["post_count"].nunique() > 1
    ):
        lowess = sm.nonparametric.lowess
        smoothed = lowess(
            filtered["avg_engagement_per_1k"],
            filtered["post_count"],
            frac=0.6
        )
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="black")

    ax.set_title(f"{ptype} Posts", fontsize=13, fontweight="bold")
    ax.set_xlabel("Posts per Week", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))
    if ax == axes[0]:
        ax.set_ylabel("Avg Engagement per Post (per 1k Followers)", fontsize=11, fontweight="bold")
    else:
        ax.set_ylabel("")


fig.suptitle("Rijksmuseum: Posting Frequency vs. Engagement per Post (Normalised) by Post Type", fontsize=15, fontweight="bold")

# Table data preparation and plotting
table_data = []
for ptype in post_types:
    df_ptype = df_filtered[df_filtered["post_type"] == ptype]
    weekly_summary = df_ptype.groupby("week").agg(
        post_count=("uri", "count"),
        total_engagement=("engagement_score", "sum"),
        avg_engagement_per_1k=("engagement_per_1k", "mean"),
        max_engagement=("engagement_per_1k", "max"),
        min_engagement=("engagement_per_1k", "min"),
    ).reset_index()

    total_posts = weekly_summary["post_count"].sum()
    total_engagement = weekly_summary["total_engagement"].sum()

    table_data.append([
        ptype,
        len(weekly_summary),
        round(weekly_summary["post_count"].mean(), 2),
        round(weekly_summary["avg_engagement_per_1k"].mean(), 2),
        round(weekly_summary["max_engagement"].max(), 2),
        round(weekly_summary["min_engagement"].min(), 2),
        format_thousands(total_posts, None),
        format_thousands(total_engagement, None)
    ])

col_labels = [
    "Post\nType",
    "Weeks\nObserved",
    "Avg\nPosts per Week",
    "Avg Engagement\nper Post (per 1k Followers)",
    "Max Engagement\nper Post",
    "Min Engagement\nper Post",
    "Total\nPosts",
    "Total\nEngagement"
]

table = table_ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    colColours=["lightgrey"] * len(col_labels)
)

# Format table header and font
header_row = 0
header_height = 0.4
for col in range(len(col_labels)):
    table[(header_row, col)].set_height(header_height)

table.auto_set_font_size(False)
table.set_fontsize(TABLE_FONT_SIZE)
table.scale(1.2, 1.2)

plt.show()

# PM_ReplyType_PostDistribution
#Load data
df = pd.read_csv(posts_latest, parse_dates=["created_at"])

# Filter reposts out and replies only (non-empty reply_type)
df = df[df["post_type"] != "Repost"].copy()
df_replies = df[df["reply_type"].notna() & (df["reply_type"] != "")].copy()

# Pie chart colours: assign a colour per reply_type for consistency
unique_reply_types = df_replies["reply_type"].unique()
# Generate a distinct palette for reply types or use pastel and map:
reply_type_colors = sns.color_palette("pastel", len(unique_reply_types))
reply_type_palette = dict(zip(unique_reply_types, reply_type_colors))

# Reply counts and percentages for pie chart
reply_counts = df_replies["reply_type"].value_counts()
total_replies = reply_counts.sum()
reply_percentages = reply_counts / total_replies * 100
labels = reply_counts.index.tolist()

# Define pie chart colours
base_pie_colors = [
    '#799F79',   # green
    '#B4B3B3',   # grey
    '#87ceeb',   # blue
    '#877B9E',   # purple
    '#924F30'    # brown
]

# Get reply types labels
labels = reply_counts.index.tolist()

# Assign colours cycling through the palette if needed
colors = [base_pie_colors[i % len(base_pie_colors)] for i in range(len(labels))]

# Then plot pie chart with these colours
plt.figure(figsize=(6, 6))
plt.pie(
    reply_counts.values,
    labels=labels,
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
    textprops={"fontsize": 10, "fontweight": "bold"}
)
plt.title("Rijksmuseum: Distribution of Reply Types", fontsize=13, fontweight="bold")

# Prepare table data for pie chart
table_data = [
    [label, f"{reply_counts[label]}", f"{reply_percentages[label]:.2f}%"]
    for label in labels
]

plt.table(
    cellText=table_data,
    colLabels=["Reply Type", "Count", "%"],
    loc="bottom",
    cellLoc="center",
    colColours=["lightgrey"] * 3,
    bbox=[0, -0.4, 1, 0.25]
)
plt.tight_layout()
plt.show()

# PM_Avg_Engagement_ReplyType
# Calculate follower count approx (reverse from mean engagement per 1k followers)
follower_count = df["engagement_per_1k_followers"].mean() * 1000

# Calculate engagement metrics per 1k followers safely on a copy to avoid warnings
df_replies = df_replies.copy()
df_replies["likes_per_1k"] = df_replies["likes_count"] / (follower_count / 1000)
df_replies["reposts_per_1k"] = df_replies["reposts_count"] / (follower_count / 1000)
df_replies["replies_per_1k"] = df_replies["replies_count"] / (follower_count / 1000)

metrics = {
    "likes_per_1k": "Likes",
    "reposts_per_1k": "Reposts",
    "replies_per_1k": "Replies"
}

# Aggregate average engagement per reply type
engagement_avg = df_replies.groupby("reply_type")[list(metrics.keys())].mean().reset_index()

# Melt for seaborn barplot
engagement_melted = engagement_avg.melt(id_vars="reply_type", var_name="metric", value_name="mean_value")
engagement_melted["metric"] = engagement_melted["metric"].map(metrics)

# Colour palette for engagement metrics (bar chart)
palette = {
    'avg_replies_norm': '#799F79',   # green
    'avg_reposts_norm': '#B4B3B3',   # grey
    'avg_likes_norm': '#87ceeb'      # blue
}

# Plot bar chart with custom palette
metric_palette_map = {
    "Likes": palette['avg_likes_norm'],
    "Reposts": palette['avg_reposts_norm'],
    "Replies": palette['avg_replies_norm']
}

plt.figure(figsize=(10, 5))
sns.barplot(
    data=engagement_melted,
    x="reply_type",
    y="mean_value",
    hue="metric",
    palette=metric_palette_map
)
plt.title("Rijksmuseum: Average Engagement per Post (per 1k Followers) by Reply Type", fontsize=13, fontweight="bold")
plt.ylabel("Engagement per Post (per 1k Followers)", fontweight="bold")
plt.xlabel("Reply Type", fontweight="bold")
plt.legend(title="Metric")
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_thousands))

# Prepare table data for engagement metrics
table_data_engagement = []
for _, row in engagement_avg.iterrows():
    table_data_engagement.append([
        row["reply_type"],
        f"{row['likes_per_1k']:.2f}",
        f"{row['reposts_per_1k']:.2f}",
        f"{row['replies_per_1k']:.2f}"
    ])

plt.table(
    cellText=table_data_engagement,
    colLabels=["Reply Type", "Likes", "Reposts", "Replies"],
    loc="bottom",
    cellLoc="center",
    colColours=["lightgrey"] * 4,
    bbox=[0, -0.4, 1, 0.3]
)
plt.tight_layout()
plt.show()

# DH_WordCloud
# Load data
df = pd.read_csv(posts_latest)

# Define colours by post type
color_dict = {
    "Original": "#87ceeb",  # Blue
    "Reply": "#799F79",     # Green
    "Quote": "#877B9E",     # Purple
    "Repost": "#B4B3B3"     # Grey
}

# Combine NLTK stopwords + WordCloud's + custom stopwords
nltk_sw = set(stopwords.words('english'))
wc_sw = set(STOPWORDS)
custom_sw = {"us", "thank", "thanks", "also", "would", "could", "please"}
all_stopwords = nltk_sw.union(wc_sw).union(custom_sw)

text_col = 'text'

def clean_text(text):
    if pd.isna(text):
        return ""
    # Remove placeholders like (@mention), (self_link) with optional trailing punctuation
    text = re.sub(r'\(?(@mention|self_link|platform_link|external_link)[\)\.,;:]*', '', text, flags=re.IGNORECASE)
    # Remove possessive 's
    text = re.sub(r"\b(\w+)'s\b", r"\1", text)
    # Tokenize and remove stopwords
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [t for t in tokens if t not in all_stopwords]
    return ' '.join(tokens)

# Prepare cleaned text by post type
texts = {}
post_types = ["Original", "Reply"]

for pt in post_types:
    texts[pt] = clean_text(' '.join(df.loc[df['post_type'] == pt, text_col].dropna().astype(str)))

# WordCloud parameters
wc_params = {
    'width': 800,
    'height': 400,
    'background_color': 'white',
    'max_words': 200,
}

def plot_wordcloud(text, title, color):
    wc = WordCloud(**wc_params, stopwords=all_stopwords).generate(text)
    plt.figure(figsize=(10, 5))
    plt.title(title, fontsize=16, fontweight='bold')
    plt.imshow(wc.recolor(color_func=lambda *args, **kwargs: color), interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot separate word clouds
for pt in post_types:
    if texts[pt].strip():
        plot_wordcloud(texts[pt], f"Rijksmuseum: {pt} Posts Word Cloud", color_dict[pt])
    else:
        print(f"No text available for post type: {pt}")

# Create frequency dictionaries for combined cloud
freq_dicts = {pt: WordCloud(stopwords=all_stopwords).process_text(texts[pt]) for pt in post_types}

# Combine frequencies, assign dominant post type per word
combined_freq = {}
word_type = {}

all_words = set(word for freq in freq_dicts.values() for word in freq.keys())

for word in all_words:
    freqs = {pt: freq_dicts[pt].get(word, 0) for pt in post_types}
    dominant_pt = max(freqs, key=freqs.get)
    combined_freq[word] = freqs[dominant_pt]
    word_type[word] = dominant_pt

class GroupColorFunc:
    def __init__(self, word_type_map, color_map):
        self.word_type_map = word_type_map
        self.color_map = color_map
    def __call__(self, word, **kwargs):
        return self.color_map.get(self.word_type_map.get(word, "Original"), "#000000")

# Generate combined word cloud
wc_combined = WordCloud(**wc_params, stopwords=all_stopwords).generate_from_frequencies(combined_freq)

plt.figure(figsize=(14, 6))
plt.imshow(wc_combined.recolor(color_func=GroupColorFunc(word_type, color_dict)), interpolation='bilinear')
plt.axis('off')
plt.title("Rijksmuseum: Mixed Word Cloud by Post Type", fontsize=16, fontweight='bold')

# Legend
patches = [Patch(color=color_dict[pt], label=pt) for pt in post_types]
plt.legend(handles=patches, title="Post Type", fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()