# Netflix Content Trends Analysis (Robust, Error-free, Dark Netflix-style Visuals)
# Copy this into VS Code as netflix_content_trends.py and run:
# python netflix_content_trends.py
#
# Requirements:
# pip install pandas matplotlib seaborn numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math

warnings.filterwarnings("ignore")

# ---------------------------
# Configuration / Parameters
# ---------------------------
DATAFILE = "Netflix Dataset.csv"    # put the CSV in same folder
TOP_COUNTRIES = 10                  # for country bar charts (use 5 or 10)
TOP_GENRES = 5                      # for top genre charts & stacked trend
DURATION_OUTLIER_IQR = 1.5         # multiplier for IQR rule
DARK_BG = True                      # dark theme on/off
NETFLIX_RED = "#e50914"             # Netflix red accent

# ---------------------------
# Styling
# ---------------------------
if DARK_BG:
    plt.style.use("dark_background")
else:
    plt.style.use("default")
sns.set_context("talk")
sns.set_palette("Reds_r")

# Helper to set common title style
def title(t):
    plt.title(t, fontsize=16, color=NETFLIX_RED, pad=14)

# ---------------------------
# Load dataset (flexible to column name variants)
# ---------------------------
df = pd.read_csv(DATAFILE)

# make a lowercase-underscore copy of column names for flexible access
orig_columns = df.columns.tolist()
colmap = {c: c.strip() for c in orig_columns}
df = df.rename(columns={c: c.strip() for c in orig_columns})

# create a normalized column-key map (lower -> actual)
col_lower_to_actual = {c.lower(): c for c in df.columns}

# helper to get column by common names with fallback
def get_col(*candidates, required=False):
    """
    Return the dataframe column name that matches one of the candidates (case-insensitive).
    If required=True and none found, raise KeyError.
    """
    for cand in candidates:
        if cand is None:
            continue
        key = cand.lower()
        if key in col_lower_to_actual:
            return col_lower_to_actual[key]
    if required:
        raise KeyError("None of the candidate column names found: {}".format(candidates))
    return None

# Typical columns observed in your file:
# Show_Id, Category, Title, Director, Cast, Country, Release_Date, Rating, Duration, Type, Description
col_category = get_col("category", "Category", required=True)        # Movie / TV Show label
col_title    = get_col("title", "Title", required=True)
col_director = get_col("director", "Director")
col_cast     = get_col("cast", "Cast")
col_country  = get_col("country", "Country")
col_release  = get_col("release_date", "Release_Date", "Release Date", "Release_Date")
col_rating   = get_col("rating", "Rating")
col_duration = get_col("duration", "Duration")
# 'Type' in user CSV appears to hold genre/listed_in
col_listed_in = get_col("type", "Type", "listed_in", "Listed_in", "Genres", "genres")

# Make working copy
df_work = df.copy()

# ---------------------------
# Basic cleaning & normalization
# ---------------------------
# Trim string columns
for c in df_work.columns:
    if df_work[c].dtype == object:
        df_work[c] = df_work[c].astype(str).str.strip()

# Replace empty-like strings with NaN for important columns
for c in [col_director, col_cast, col_country, col_release, col_rating, col_listed_in]:
    if c is not None and c in df_work.columns:
        df_work.loc[df_work[c].str.lower().isin(["", "unknown", "nan", "none", "n/a"]), c] = np.nan

# ---------------------------
# Parse Release Date -> year_released
# ---------------------------
if col_release is not None:
    df_work["release_date_parsed"] = pd.to_datetime(df_work[col_release], errors="coerce", infer_datetime_format=True)
    df_work["year_released"] = df_work["release_date_parsed"].dt.year
else:
    df_work["release_date_parsed"] = pd.NaT
    df_work["year_released"] = np.nan

# Also create year_added if a 'date_added' exists (some datasets)
col_date_added = get_col("date_added", "Date_Added", "added_date")
if col_date_added:
    df_work["date_added_parsed"] = pd.to_datetime(df_work[col_date_added], errors="coerce", infer_datetime_format=True)
    df_work["year_added"] = df_work["date_added_parsed"].dt.year
else:
    df_work["year_added"] = df_work["year_released"]

# ---------------------------
# Parse Duration: extract minutes for movies; seasons for TV shows (if present)
# ---------------------------
df_work["duration_minutes"] = np.nan
df_work["duration_seasons"] = np.nan

if col_duration:
    def parse_duration(v):
        if pd.isna(v):
            return (np.nan, np.nan)
        s = str(v).lower()
        if "min" in s:
            nums = "".join(ch if ch.isdigit() else " " for ch in s).split()
            for n in nums:
                if n.isdigit():
                    return (int(n), np.nan)
        if "season" in s:
            nums = "".join(ch if ch.isdigit() else " " for ch in s).split()
            for n in nums:
                if n.isdigit():
                    return (np.nan, int(n))
        import re
        m = re.search(r"(\d+)", s)
        if m:
            return (int(m.group(1)), np.nan)
        return (np.nan, np.nan)

    parsed = df_work[col_duration].apply(parse_duration)
    df_work["duration_minutes"] = parsed.apply(lambda x: x[0])
    df_work["duration_seasons"] = parsed.apply(lambda x: x[1])

# ---------------------------
# Genres: explode into one-genre-per-row
# ---------------------------
if col_listed_in:
    df_work["_genres_raw"] = df_work[col_listed_in].where(df_work[col_listed_in].notna(), "")
    def split_genres(s):
        if not s or pd.isna(s):
            return []
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return parts
    df_work["_genres_list"] = df_work["_genres_raw"].apply(split_genres)
else:
    df_work["_genres_list"] = [[] for _ in range(len(df_work))]

df_genres = df_work.loc[:, [col_title, col_category, "year_released", "year_added", col_country, "_genres_list"]].explode("_genres_list")
df_genres = df_genres.rename(columns={"_genres_list": "genre"})
df_genres["genre"] = df_genres["genre"].astype(str).str.strip()
df_genres.loc[df_genres["genre"] == "", "genre"] = np.nan
df_genres = df_genres[df_genres["genre"].notna()]

# ---------------------------
# Remove extreme duration outliers for movie visuals (IQR method)
# ---------------------------
movies_mask = df_work[col_category].str.lower() == "movie" if col_category else (df_work[col_category].astype(str).str.lower() == "movie")
movie_minutes = df_work.loc[movies_mask, "duration_minutes"].dropna().astype(float)

if len(movie_minutes) > 0:
    q1 = movie_minutes.quantile(0.25)
    q3 = movie_minutes.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - DURATION_OUTLIER_IQR * iqr
    upper = q3 + DURATION_OUTLIER_IQR * iqr
    df_work["duration_outlier"] = False
    cond = (df_work["duration_minutes"].notna()) & ((df_work["duration_minutes"].astype(float) < lower) | (df_work["duration_minutes"].astype(float) > upper))
    df_work.loc[cond, "duration_outlier"] = True
else:
    df_work["duration_outlier"] = False

df_clean = df_work[~df_work["duration_outlier"]].copy()

# ---------------------------
# Utility: ensure numeric years and sensible ranges for plotting
# ---------------------------
def clean_year_series(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.dropna().astype(int)
    s = s[(s > 1900) & (s <= pd.Timestamp.now().year + 1)]
    return s

df_clean["year_released_clean"] = clean_year_series(df_clean["year_released"])
df_clean["year_added_clean"] = clean_year_series(df_clean["year_added"])

# ---------------------------
# VISUAL 1: Movies vs TV Shows (overall & per year stacked)
# ---------------------------
fig, ax = plt.subplots(figsize=(8, 5))
type_counts = df_clean[col_category].value_counts()
sns.barplot(x=type_counts.values, y=type_counts.index, orient="h", ax=ax, edgecolor="white")
title("Overall: Movies vs TV Shows")
ax.set_xlabel("Number of Titles")
ax.set_ylabel("")
plt.tight_layout()
plt.show()

if "year_added_clean" in df_clean.columns and not df_clean["year_added_clean"].isna().all():
    pivot = df_clean.groupby(["year_added_clean", col_category]).size().unstack(fill_value=0)
    pivot = pivot.sort_index()
    years = pivot.index.values
    labels = pivot.columns.tolist()
    values = [pivot[c].values for c in labels]
    
    # Define colors: red for Movies, orange for TV Shows (adjust according to column order)
    colors = []
    for lbl in labels:
        if lbl.lower() == "movie":
            colors.append("#e50914")   # Netflix Red
        else:
            colors.append("#ff8c00")   # Orange
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(years, values, labels=labels, colors=colors)
    title("Yearly Additions: Movies vs TV Shows")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Titles Added")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


# ---------------------------
# VISUAL 2: Top Genres (bar) and Top Genres Trend (stacked area)
# ---------------------------
# Top genres with different colors per bar
genre_counts = df_genres["genre"].value_counts().head(max(TOP_GENRES, 10))
fig, ax = plt.subplots(figsize=(10, 6))

# Create a color palette with as many colors as bars
colors = sns.color_palette("hls", len(genre_counts))  # "hls" gives distinct colors
sns.barplot(x=genre_counts.values, y=genre_counts.index, orient="h", ax=ax, palette=colors, edgecolor="white")

title("Top {} Genres (overall)".format(TOP_GENRES))
ax.set_xlabel("Number of Titles")
plt.tight_layout()
plt.show()


topK = df_genres["genre"].value_counts().head(TOP_GENRES).index.tolist()
genre_yearly = df_genres[df_genres["genre"].isin(topK)].groupby(["year_added", "genre"]).size().unstack(fill_value=0)
genre_yearly = genre_yearly.loc[genre_yearly.index.notna()]
genre_yearly = genre_yearly.sort_index()

if len(genre_yearly) > 0:
    years = genre_yearly.index.astype(int).values
    series = [genre_yearly[g].values for g in genre_yearly.columns]

    # Generate distinct colors for each genre
    colors = sns.color_palette("tab10", len(genre_yearly.columns))  # "tab10" gives up to 10 distinct colors

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(years, series, labels=genre_yearly.columns, colors=colors)
    title("Yearly Trend: Top {} Genres".format(TOP_GENRES))
    ax.set_xlabel("Year Added")
    ax.set_ylabel("Number of Titles")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


# ---------------------------
# VISUAL 3: Top Countries (bar + heatmap)
# ---------------------------
if col_country:
    country_counts = df_clean[col_country].dropna().value_counts().head(TOP_COUNTRIES)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Map of country to main flag colors (approximation)
    flag_colors = {
        "United States": "#b22234",  # Red
        "India": "#ff9933",           # Saffron
        "United Kingdom": "#00247d",  # Dark Blue
        "Canada": "#ff0000",          # Red
        "Australia": "#00008b",       # Blue
        "Japan": "#bc002d",           # Red
        "Germany": "#ffce00",         # Yellow
        "France": "#0055a4",          # Blue
        "South Korea": "#003478",     # Blue
        "Mexico": "#006847",          # Green
    }

    # Create color list for the bars, fallback to grey if not in map
    colors = [flag_colors.get(country, "#808080") for country in country_counts.index]

    sns.barplot(x=country_counts.values, y=country_counts.index, orient="h", ax=ax, palette=colors, edgecolor="white")
    title("Top {} Countries by # of Titles".format(TOP_COUNTRIES))
    ax.set_xlabel("Number of Titles")

    # Add numeric labels at the end of each bar
    for i, v in enumerate(country_counts.values):
        ax.text(v + max(country_counts.values)*0.01, i, str(v), color="white", va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


    # Heatmap: genres vs countries
    top_countries_list = df_clean[col_country].dropna().value_counts().head(8).index.tolist()
    top_genres_list = df_genres["genre"].value_counts().head(8).index.tolist()
    heat_df = df_genres[df_genres[col_country].isin(top_countries_list) & df_genres["genre"].isin(top_genres_list)]
    if not heat_df.empty:
        pivot_heat = heat_df.groupby([col_country, "genre"]).size().unstack(fill_value=0)
        pivot_norm = pivot_heat.div(pivot_heat.sum(axis=1).replace(0, 1), axis=0)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_norm, annot=True, fmt=".2f", linewidths=0.5, ax=ax, cbar_kws={"label": "Proportion"})
        title("Genre Proportions across Top Countries (normalized)")
        ax.set_xlabel("Genre")
        ax.set_ylabel("Country")
        plt.tight_layout()
        plt.show()

# ---------------------------
# VISUAL 4: Duration Comparison — Movies vs TV Shows (outliers removed)
# ---------------------------
movies_df = df_clean[df_clean[col_category].str.lower() == "movie"]
tv_df = df_clean[df_clean[col_category].str.lower() == "tv show"]

# Remove outliers for movies using IQR method
movie_durations = movies_df["duration_minutes"].dropna()
if len(movie_durations) > 0:
    q1 = movie_durations.quantile(0.25)
    q3 = movie_durations.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - DURATION_OUTLIER_IQR * iqr
    upper = q3 + DURATION_OUTLIER_IQR * iqr
    movies_df = movies_df[(movies_df["duration_minutes"] >= lower) & (movies_df["duration_minutes"] <= upper)]

# Optionally remove extreme TV show seasons (top 1% cutoff)
tv_seasons = tv_df["duration_seasons"].dropna()
if len(tv_seasons) > 0:
    upper_tv = tv_seasons.quantile(0.99)
    tv_df = tv_df[tv_df["duration_seasons"] <= upper_tv]

# Boxplot comparison
fig, ax = plt.subplots(figsize=(10, 6))
duration_data = [
    movies_df["duration_minutes"].dropna(),
    tv_df["duration_seasons"].dropna()
]
labels = ["Movies (Minutes)", "TV Shows (Seasons)"]
sns.boxplot(data=duration_data, ax=ax)
ax.set_xticklabels(labels)
title("Content Duration Comparison: Movies vs TV Shows (Outliers Removed)")
ax.set_ylabel("Duration Value (Minutes or Seasons)")
plt.tight_layout()
plt.show()

# Histogram: Movies Duration Only
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(movies_df["duration_minutes"].dropna(), bins=25, kde=True, ax=ax, color="#e50914")
ax.set_title("Movies Duration Distribution", color=NETFLIX_RED)
ax.set_xlabel("Duration (Minutes)")
plt.suptitle("Movie Duration Distribution", color=NETFLIX_RED, fontsize=16)
plt.tight_layout()
plt.show()


# ---------------------------
# Final printed insights
# ---------------------------
print("\n\n===== QUICK ACTIONABLE INSIGHTS (AUTO-GENERATED) =====\n")
top_genres_overall = df_genres["genre"].value_counts().head(5)
print("Top genres overall:")
for g, c in top_genres_overall.items():
    print(" - {:30s} {:6d} titles".format(g, int(c)))

if col_country:
    top_ct = df_clean[col_country].dropna().value_counts().head(10)
    print("\nTop countries (by title count):")
    for c, v in top_ct.items():
        print(" - {:20s} {:6d} titles".format(str(c)[:20], int(v)))

if "year_added_clean" in df_clean.columns and not df_clean["year_added_clean"].isna().all():
    yearly_added = df_clean["year_added_clean"].value_counts().sort_index()
    if len(yearly_added) > 1:
        years = yearly_added.index.tolist()
        last = years[-1]
        prev = years[-2]
        last_val = int(yearly_added.loc[last])
        prev_val = int(yearly_added.loc[prev])
        growth = (last_val - prev_val) / prev_val * 100 if prev_val != 0 else math.nan
        print("\nRecent additions growth: {} -> {} : {:.1f}% change".format(prev, last, growth))
    else:
        print("\nNot enough yearly data to compute recent growth.")

print("\nScript finished.")
