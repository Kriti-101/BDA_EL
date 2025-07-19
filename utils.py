# utils.py - Complete Implementation with All Graphs
import folium
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import json # <--- ADD THIS IMPORT for JSON parsing

# Ensure this is included for data cleaning (already there)
from pyspark.sql.functions import col


def load_data(file_path):
    """Loads and preprocesses the BDT.csv file."""
    df = pd.read_csv(file_path)

    # Convert month columns to numeric if needed
    month_cols = [f'i_mid_{m}' for m in ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec']]
    for col in month_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')

    return df

def get_variable_explanations():
    """Returns detailed explanations of all variables."""
    return """
    ### Variable Explanations:

    **Coordinates:**
    - **X:** Longitude coordinate
    - **Y:** Latitude coordinate

    **Concentration Estimates:**
    - **i_mid:** Mid-estimate of microplastic concentration (particles/km²)
    - **i_low:** Low-estimate of microplastic concentration
    - **i_high:** High-estimate of microplastic concentration

    **Monthly Data:**
    - **i_mid_jan to i_mid_dec:** Monthly mid-estimates
    - **i_low_jan to i_low_dec:** Monthly low-estimates
    - **i_high_jan to i_high_dec:** Monthly high-estimates
    - **runoff_jan to runoff_dec:** Monthly runoff estimates (m³/s)

    **Additional Metrics:**
    - **mpw:** Microplastic weight (kg)
    - **area:** Area of sampling location (km²)

    **Significance:**
    These variables enable comprehensive analysis of spatial distribution, seasonal patterns,
    pollution sources, and potential environmental impacts of microplastics.
    """

# ======== Graph Functions ========

def spatial_analysis_colab(df, sample_size=1000):
    # Get the number of rows in the DataFrame
    row_count = len(df)
    st.write("Total rows in DataFrame:", row_count)

    if row_count == 0:
        raise ValueError("Input DataFrame is empty.")

    fraction = min(1.0, sample_size / row_count)
    df_sample = df.sample(frac=fraction)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df_sample['X'], df_sample['Y'],
                          c=df_sample['i_mid'], cmap='viridis', alpha=0.6, s=20)
    plt.xlabel('Longitude (X)')
    plt.ylabel('Latitude (Y)')
    plt.title('Spatial Distribution of Microplastic Concentrations')
    plt.colorbar(scatter, label='Concentration (i_mid)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    st.pyplot(plt)

    return df_sample

def plot_total_monthly_input(df):
    """Plots total monthly plastic input (mid estimate)"""
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    monthly_totals = df[[f'i_mid_{m}' for m in months]].sum().reset_index()
    monthly_totals.columns = ['Month', 'Total']
    monthly_totals['Month'] = monthly_totals['Month'].str.replace('i_mid_', '').str.title()

    plt.figure(figsize=(14, 7))
    sns.barplot(x='Month', y='Total', data=monthly_totals, palette='viridis')
    plt.title('Total Monthly Plastic Input (Mid-Estimate)')
    plt.xlabel('Month')
    plt.ylabel('Plastic Input (tonnes)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    st.pyplot(plt)

    return monthly_totals

def plot_top_rivers(df):
    """Creates an interactive map of the top 20 polluting rivers."""
    if 'X' not in df.columns or 'Y' not in df.columns or 'i_mid' not in df.columns:
        st.error("DataFrame must contain 'X', 'Y', and 'i_mid' columns.")
        return None

    df['i_mid'] = pd.to_numeric(df['i_mid'], errors='coerce')
    df.dropna(subset=['i_mid', 'X', 'Y'], inplace=True)

    if len(df) < 20:
        st.warning(f"Less than 20 valid data points available ({len(df)}). Displaying all available for top rivers map.")
        top_rivers = df.nlargest(len(df), 'i_mid')[['X', 'Y', 'i_mid']]
    else:
        top_rivers = df.nlargest(20, 'i_mid')[['X', 'Y', 'i_mid']]

    if top_rivers.empty:
        st.info("No valid data available for the top rivers to plot on map after filtering.")
        return None

    map_center_y = top_rivers['Y'].mean() if not top_rivers['Y'].isnull().all() else 0
    map_center_x = top_rivers['X'].mean() if not top_rivers['X'].isnull().all() else 0
    map_center = [map_center_y, map_center_x]

    zoom_start = 3
    if not top_rivers.empty:
        min_lat, max_lat = top_rivers['Y'].min(), top_rivers['Y'].max()
        min_lon, max_lon = top_rivers['X'].min(), top_rivers['X'].max()
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        if lat_range < 5 and lon_range < 5:
            zoom_start = 8
        elif lat_range < 10 and lon_range < 10:
            zoom_start = 6
        elif lat_range > 50 or lon_range > 50:
            zoom_start = 2

    pollution_map = folium.Map(location=map_center, zoom_start=zoom_start)

    max_i_mid = top_rivers['i_mid'].max()
    radius_scale_factor = 5000
    if max_i_mid > 0:
        if max_i_mid < 10:
            radius_scale_factor = 0.5
        elif max_i_mid < 100:
            radius_scale_factor = 5
        elif max_i_mid < 1000:
            radius_scale_factor = 50
        elif max_i_mid < 10000:
            radius_scale_factor = 500

    for index, row in top_rivers.iterrows():
        if pd.isna(row['Y']) or pd.isna(row['X']) or pd.isna(row['i_mid']):
            continue

        radius = max(0.5, row['i_mid'] / radius_scale_factor)

        folium.CircleMarker(
            location=[row['Y'], row['X']],
            radius=radius,
            color='red',
            fill=True,
            fill_opacity=0.6,
            popup=f"Pollution: {row['i_mid']:.2f} particles/km²"
        ).add_to(pollution_map)

    if not top_rivers.empty:
        pollution_map.fit_bounds(pollution_map.get_bounds())

    return pollution_map

def plot_plastic_heatmap_folium(df):
    """Creates a Folium heatmap of plastic concentration (replaces Plotly version)."""
    if 'X' not in df.columns or 'Y' not in df.columns or 'i_mid' not in df.columns:
        st.error("DataFrame must contain 'X', 'Y', and 'i_mid' columns for heatmap.")
        return None

    df_clean = df.dropna(subset=['X', 'Y', 'i_mid'])
    if df_clean.empty:
        st.warning("No valid data available for heatmap after filtering.")
        return None

    heat_data = [
        [row['Y'], row['X'], row['i_mid']]
        for _, row in df_clean.iterrows()
        if row['i_mid'] > 0
    ]

    if not heat_data:
        st.warning("No positive 'i_mid' values to plot in heatmap.")
        return None

    map_center = [df_clean['Y'].mean(), df_clean['X'].mean()] if not df_clean.empty else [30, 0]
    heatmap = folium.Map(
        location=map_center,
        zoom_start=2,
        tiles="CartoDB positron"
    )

    HeatMap(
        heat_data,
        radius=12,
        blur=15,
        gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'},
        max_zoom=10
    ).add_to(heatmap)

    return heatmap


def plot_top_rivers_by_month(pandas_df):
       """Creates an animated bar chart of the top river plastic input by month."""
       cleaned_df = pandas_df[
           ~((pandas_df["i_mid"] == 0) & (pandas_df["i_low"] == 0) & (pandas_df["i_high"] == 0))
       ]

       months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

       melted_df = cleaned_df.melt(
           id_vars=['X', 'Y', 'i_mid', 'mpw', 'area'],
           value_vars=[f'i_mid_{m}' for m in months],
           var_name='month',
           value_name='monthly_input'
       )

       melted_df['month'] = melted_df['month'].str.replace('i_mid_', '')

       melted_df['river_id'] = melted_df['X'].astype(str) + '_' + melted_df['Y'].astype(str)

       top_n = 20
       top_rivers_by_month = (
           melted_df.groupby('month')
           .apply(lambda x: x.nlargest(top_n, 'monthly_input'))
           .reset_index(drop=True)
       )

       month_order = {m: i for i, m in enumerate(months)}
       top_rivers_by_month['month_index'] = top_rivers_by_month['month'].map(month_order)
       top_rivers_by_month.sort_values('month_index', inplace=True)

       fig = px.bar(
           top_rivers_by_month,
           x='monthly_input',
           y='river_id',
           animation_frame='month',
           animation_group='river_id',
           color='mpw',
           range_x=[0, top_rivers_by_month['monthly_input'].max() * 1.1],
           labels={'monthly_input': 'Monthly Plastic Input (tonnes)', 'river_id': 'River (X_Y)'},
           title='Top River Plastic Input by Month'
       )
       fig.update_layout(height=800, width=1200)

       return fig


def plot_correlational_analysis(df):
    """Creates correlation matrix heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title='Correlational Analysis Between Variables'
    )
    return fig

# NEW FUNCTION to load hotspot data
def load_hotspot_data(file_path="hotspot_analysis.txt"):
    """Loads and parses hotspot data from the specified text file."""
    hotspot_records = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                # Split by tab, take the second part (JSON string)
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    st.warning(f"Skipping malformed line in hotspot_analysis.txt: {line.strip()}")
                    continue
                try:
                    data = json.loads(parts[1])
                    for coord_str in data["sample_locations"]:
                        lat, lon = map(float, coord_str.split(","))
                        hotspot_records.append({
                            "latitude": lon,  # Note: Your original code had lat=lon, lon=lat. Keeping that.
                            "longitude": lat,
                            "hotspot_score": data["hotspot_score"],
                            "grid": data["grid"]
                        })
                except json.JSONDecodeError as e:
                    st.warning(f"JSON decoding error in hotspot_analysis.txt line: {line.strip()} - {e}")
                except ValueError as e:
                    st.warning(f"Coordinate parsing error in hotspot_analysis.txt line: {line.strip()} - {e}")
    except FileNotFoundError:
        st.error(f"Error: Hotspot analysis file not found at {file_path}. Please ensure it exists.")
        return pd.DataFrame()  # Return empty DataFrame on error
    except Exception as e:
        st.error(f"An unexpected error occurred while loading hotspot data: {e}")
        return pd.DataFrame()

    return pd.DataFrame(hotspot_records)


def plot_hotspot_analysis(hotspot_df):
    """Creates the Hotspot Scatter Plot on OpenStreetMap using the provided DataFrame."""
    if hotspot_df.empty:
        st.warning("Hotspot data is empty. Cannot generate plot.")
        return go.Figure() # Return an empty Plotly figure

    fig = px.scatter_mapbox(
        hotspot_df,
        lat="latitude",
        lon="longitude",
        color="hotspot_score",
        size="hotspot_score",
        hover_name="grid",
        color_continuous_scale="Inferno",
        size_max=10,
        zoom=1,
        height=600
    )
    fig.update_layout(mapbox_style="open-street-map", title="Hotspot Scatter Plot on OpenStreetMap")
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def plot_seasonal_analysis(df):
    """Analyzes seasonal patterns in microplastic concentration"""
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
             'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    monthly_data = df[[f'i_mid_{m}' for m in months]]
    monthly_data.columns = [m.capitalize() for m in months]

    fig = px.box(
        monthly_data,
        title="Seasonal Variation in Microplastic Concentration",
        labels={'value': 'Concentration', 'variable': 'Month'}
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Concentration (particles/km²)"
    )
    return fig

def plot_severity_analysis(df):
    """Assesses severity based on concentration-area relationship"""
    fig = px.scatter(
        df,
        x="area",
        y="i_high",
        color="runoff_jul",
        size="mpw",
        log_x=True,
        log_y=True,
        title="Severity Analysis: High Concentration vs Area & Runoff",
        labels={
            "i_high": "High Estimate Concentration",
            "area": "Area (log scale)",
            "runoff_jul": "July Runoff",
            "mpw": "Microplastic Weight"
        }
    )
    return fig

