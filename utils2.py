# utils.py - Complete Enhanced Implementation with Dynamic and Animated Graphs
import folium
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from streamlit_folium import folium_static
from folium.plugins import HeatMap, MarkerCluster, TimestampedGeoJson
import json
import plotly.figure_factory as ff
from datetime import datetime, timedelta
from pyspark.sql.functions import col

def load_data(file_path):
    """Loads and preprocesses the BDT.csv file with enhanced error handling."""
    try:
        df = pd.read_csv(file_path)
        
        # Convert month columns to numeric if needed
        month_cols = [f'i_mid_{m}' for m in ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                            'jul', 'aug', 'sep', 'oct', 'nov', 'dec']]
        for col in month_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def get_variable_explanations():
    """Returns enhanced variable explanations with formatting."""
    return """
    ### 📊 Variable Explanations:

    **🌐 Geographic Coordinates:**
    - **X:** Longitude coordinate (decimal degrees)
    - **Y:** Latitude coordinate (decimal degrees)

    **🔬 Concentration Estimates:**
    - **i_mid:** Mid-estimate of microplastic concentration (particles/km²)
    - **i_low:** Conservative low-estimate of concentration
    - **i_high:** Maximum high-estimate of concentration

    **📅 Temporal Data:**
    - **i_mid_jan to i_mid_dec:** Monthly concentration mid-estimates
    - **i_low_jan to i_low_dec:** Monthly conservative estimates
    - **i_high_jan to i_high_dec:** Monthly maximum estimates
    - **runoff_jan to runoff_dec:** Monthly water runoff (m³/s)

    **⚖️ Physical Metrics:**
    - **mpw:** Total microplastic weight (kg)
    - **area:** Sampling area coverage (km²)

    **🎯 Analysis Applications:**
    These variables enable comprehensive spatio-temporal analysis, pollution source identification,
    seasonal pattern detection, and environmental impact assessment of microplastic contamination.
    """

# ======== Enhanced Graph Functions ========

def spatial_analysis_colab(df, sample_size=1000):
    """Enhanced spatial analysis with 3D visualization and clustering."""
    row_count = len(df)
    st.write(f"📊 **Total rows in DataFrame:** {row_count:,}")

    if row_count == 0:
        raise ValueError("Input DataFrame is empty.")

    fraction = min(1.0, sample_size / row_count)
    df_sample = df.sample(frac=fraction)

    # Create interactive 3D scatter plot
    fig = go.Figure()

    # Add 3D scatter trace with enhanced styling
    fig.add_trace(go.Scatter3d(
        x=df_sample['X'],
        y=df_sample['Y'],
        z=df_sample['i_mid'],
        mode='markers',
        marker=dict(
            size=6,
            color=df_sample['i_mid'],
            colorscale='Plasma',
            opacity=0.8,
            colorbar=dict(title="Concentration<br>(particles/km²)", thickness=15),
            line=dict(width=0.5, color='white')
        ),
        text=[f'Lat: {y:.2f}<br>Lon: {x:.2f}<br>Conc: {c:.2f}' 
              for x, y, c in zip(df_sample['X'], df_sample['Y'], df_sample['i_mid'])],
        hovertemplate='<b>Location</b><br>%{text}<extra></extra>',
        name='Microplastic Sites'
    ))

    # Enhanced layout with dark theme
    fig.update_layout(
        title=dict(
            text='🌊 3D Spatial Distribution of Microplastic Concentrations',
            x=0.5,
            font=dict(size=18, color='#2E86C1')
        ),
        scene=dict(
            xaxis_title='Longitude (°)',
            yaxis_title='Latitude (°)',
            zaxis_title='Concentration (particles/km²)',
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.2)', showbackground=True, backgroundcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.2)', showbackground=True, backgroundcolor='rgba(0,0,0,0.1)'),
            zaxis=dict(gridcolor='rgba(255,255,255,0.2)', showbackground=True, backgroundcolor='rgba(0,0,0,0.1)'),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)
    return df_sample

def plot_total_monthly_input(df):
    """Enhanced animated monthly input visualization."""
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    monthly_totals = df[[f'i_mid_{m}' for m in months]].sum().reset_index()
    monthly_totals.columns = ['Month', 'Total']
    monthly_totals['Month'] = monthly_totals['Month'].str.replace('i_mid_', '').str.title()
    
    # Add month order for proper sequencing
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_totals['Month'] = pd.Categorical(monthly_totals['Month'], categories=month_order, ordered=True)
    monthly_totals = monthly_totals.sort_values('Month')

    # Create animated bar chart
    fig = px.bar(
        monthly_totals,
        x='Month',
        y='Total',
        color='Total',
        color_continuous_scale='Turbo',
        title='🗓️ Total Monthly Plastic Input Analysis',
        labels={'Total': 'Plastic Input (tonnes)', 'Month': 'Month'}
    )

    # Enhanced styling
    fig.update_traces(
        texttemplate='%{y:.1f}T',
        textposition='outside',
        marker_line_width=2,
        marker_line_color='white'
    )
    
    fig.update_layout(
        title=dict(x=0.5, font=dict(size=18)),
        xaxis=dict(
            tickangle=45,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
    return monthly_totals

def plot_top_rivers(df):
    """Enhanced interactive map with clustering and animations."""
    if 'X' not in df.columns or 'Y' not in df.columns or 'i_mid' not in df.columns:
        st.error("DataFrame must contain 'X', 'Y', and 'i_mid' columns.")
        return None

    df['i_mid'] = pd.to_numeric(df['i_mid'], errors='coerce')
    df.dropna(subset=['i_mid', 'X', 'Y'], inplace=True)

    if len(df) < 20:
        st.warning(f"Less than 20 valid data points available ({len(df)}). Displaying all available.")
        top_rivers = df.nlargest(len(df), 'i_mid')[['X', 'Y', 'i_mid']]
    else:
        top_rivers = df.nlargest(20, 'i_mid')[['X', 'Y', 'i_mid']]

    if top_rivers.empty:
        st.info("No valid data available for the top rivers map.")
        return None

    # Calculate map center
    map_center = [top_rivers['Y'].mean(), top_rivers['X'].mean()]
    
    # Create enhanced map with dark theme
    pollution_map = folium.Map(
        location=map_center,
        zoom_start=3,
        tiles=None
    )
    
    # Add custom dark tile layer
    folium.TileLayer(
        'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='Dark Theme',
        overlay=False,
        control=True
    ).add_to(pollution_map)

    # Add marker cluster for better performance
    marker_cluster = MarkerCluster(
        name="Pollution Hotspots",
        overlay=True,
        control=True
    ).add_to(pollution_map)

    # Create color scale based on pollution levels
    max_pollution = top_rivers['i_mid'].max()
    min_pollution = top_rivers['i_mid'].min()

    for index, row in top_rivers.iterrows():
        if pd.isna(row['Y']) or pd.isna(row['X']) or pd.isna(row['i_mid']):
            continue

        # Color intensity based on pollution level
        intensity = (row['i_mid'] - min_pollution) / (max_pollution - min_pollution)
        color = f"#{int(255 * intensity):02x}{int(255 * (1-intensity)):02x}00"
        
        # Dynamic radius based on pollution level
        radius = max(5, min(25, 5 + (row['i_mid'] / max_pollution) * 20))

        folium.CircleMarker(
            location=[row['Y'], row['X']],
            radius=radius,
            color='white',
            weight=2,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            popup=folium.Popup(f"""
                <div style='font-family: Arial; width: 200px;'>
                    <h4>🌊 Pollution Hotspot</h4>
                    <hr>
                    <b>Concentration:</b> {row['i_mid']:.2f} particles/km²<br>
                    <b>Coordinates:</b> {row['Y']:.3f}, {row['X']:.3f}<br>
                    <b>Severity:</b> {'🔴 Critical' if intensity > 0.8 else '🟡 High' if intensity > 0.5 else '🟢 Moderate'}
                </div>
            """, max_width=300)
        ).add_to(marker_cluster)

    # Add layer control
    folium.LayerControl().add_to(pollution_map)
    
    return pollution_map

def plot_plastic_heatmap_folium(df):
    """Enhanced heatmap with temporal animation capabilities."""
    if 'X' not in df.columns or 'Y' not in df.columns or 'i_mid' not in df.columns:
        st.error("DataFrame must contain 'X', 'Y', and 'i_mid' columns for heatmap.")
        return None

    df_clean = df.dropna(subset=['X', 'Y', 'i_mid'])
    if df_clean.empty:
        st.warning("No valid data available for heatmap.")
        return None

    # Prepare heat data with intensity scaling
    max_intensity = df_clean['i_mid'].quantile(0.95)  # Use 95th percentile to avoid outliers
    heat_data = [
        [row['Y'], row['X'], min(row['i_mid'] / max_intensity, 1.0)]
        for _, row in df_clean.iterrows()
        if row['i_mid'] > 0
    ]

    if not heat_data:
        st.warning("No positive concentration values for heatmap.")
        return None

    # Create enhanced map
    map_center = [df_clean['Y'].mean(), df_clean['X'].mean()]
    heatmap = folium.Map(
        location=map_center,
        zoom_start=2,
        tiles=None
    )

    # Add multiple tile layers
    folium.TileLayer('OpenStreetMap', name='Standard').add_to(heatmap)
    folium.TileLayer('Stamen Terrain', name='Terrain').add_to(heatmap)
    folium.TileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(heatmap)

    # Add enhanced heatmap with custom gradient
    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_zoom=12,
        radius=15,
        blur=20,
        gradient={
            0.0: 'navy',
            0.25: 'blue', 
            0.5: 'cyan',
            0.75: 'yellow',
            1.0: 'red'
        }
    ).add_to(heatmap)

    # Add layer control
    folium.LayerControl().add_to(heatmap)

    return heatmap

def plot_top_rivers_by_month(pandas_df):
    """Enhanced animated racing bar chart with smooth transitions."""
    cleaned_df = pandas_df[
        ~((pandas_df["i_mid"] == 0) & (pandas_df["i_low"] == 0) & (pandas_df["i_high"] == 0))
    ]

    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    melted_df = cleaned_df.melt(
        id_vars=['X', 'Y', 'i_mid', 'mpw', 'area'],
        value_vars=[f'i_mid_{m}' for m in months],
        var_name='month',
        value_name='monthly_input'
    )

    melted_df['month'] = melted_df['month'].str.replace('i_mid_', '')
    melted_df['month_name'] = melted_df['month'].map(dict(zip(months, month_names)))
    melted_df['river_id'] = 'River_' + melted_df.index.astype(str)

    # Create river names based on coordinates for better identification
    melted_df['river_display'] = (
        melted_df['X'].round(2).astype(str) + '°E, ' + 
        melted_df['Y'].round(2).astype(str) + '°N'
    )

    top_n = 15
    top_rivers_by_month = (
        melted_df.groupby('month')
        .apply(lambda x: x.nlargest(top_n, 'monthly_input'))
        .reset_index(drop=True)
    )

    # Create animated racing bar chart
    fig = px.bar(
        top_rivers_by_month,
        x='monthly_input',
        y='river_display',
        animation_frame='month_name',
        animation_group='river_id',
        color='mpw',
        color_continuous_scale='Viridis',
        range_x=[0, top_rivers_by_month['monthly_input'].max() * 1.1],
        labels={
            'monthly_input': 'Monthly Plastic Input (tonnes)', 
            'river_display': 'River Location',
            'mpw': 'Microplastic Weight (kg)'
        },
        title='🏆 Top Rivers Monthly Plastic Input Race'
    )

    # Enhanced styling with animations
    fig.update_layout(
        height=800,
        title=dict(x=0.5, font=dict(size=20)),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)',
            tickformat='.1f'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_colorbar=dict(
            title="Microplastic Weight",
            thickness=15,
            len=0.7
        )
    )

    # Smooth animation settings
    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300

    return fig

def plot_correlational_analysis(df):
    """Enhanced correlation analysis with interactive features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    # Create interactive correlation heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0,
        title='🔗 Interactive Correlation Matrix Analysis'
    )
    
    # Enhanced layout
    fig.update_layout(
        title=dict(x=0.5, font=dict(size=18)),
        height=700,
        width=800
    )
    
    fig.update_traces(
        textfont_size=10,
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
    )

    return fig

def load_hotspot_data(file_path="hotspot_analysis.txt"):
    """Enhanced hotspot data loading with error handling."""
    hotspot_records = []
    try:
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    st.warning(f"⚠️ Skipping malformed line {line_num} in hotspot_analysis.txt")
                    continue
                try:
                    data = json.loads(parts[1])
                    for coord_str in data["sample_locations"]:
                        lat, lon = map(float, coord_str.split(","))
                        hotspot_records.append({
                            "latitude": lon,  # Note: maintaining original coordinate mapping
                            "longitude": lat,
                            "hotspot_score": data["hotspot_score"],
                            "grid": data["grid"]
                        })
                except (json.JSONDecodeError, ValueError) as e:
                    st.warning(f"⚠️ Error parsing line {line_num}: {e}")
                    
    except FileNotFoundError:
        st.error(f"❌ Hotspot analysis file not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Unexpected error loading hotspot data: {e}")
        return pd.DataFrame()

    return pd.DataFrame(hotspot_records)

def plot_hotspot_analysis(hotspot_df):
    """Enhanced hotspot visualization with 3D perspective."""
    if hotspot_df.empty:
        st.warning("📊 Hotspot data is empty. Cannot generate visualization.")
        return go.Figure()

    # Create enhanced scatter mapbox
    fig = px.scatter_mapbox(
        hotspot_df,
        lat="latitude",
        lon="longitude",
        color="hotspot_score",
        size="hotspot_score",
        hover_name="grid",
        hover_data={"hotspot_score": ":.2f"},
        color_continuous_scale="plasma",
        size_max=20,
        zoom=2,
        height=700,
        title="🔥 Global Microplastic Hotspot Analysis"
    )
    
    # Enhanced styling
    fig.update_layout(
        mapbox_style="open-street-map",
        title=dict(x=0.5, font=dict(size=18)),
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Hotspot<br>Score",
            thickness=15,
            len=0.7
        )
    )
    
    return fig

def plot_seasonal_analysis(df):
    """Enhanced seasonal analysis with statistical insights."""
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
             'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    monthly_data = df[[f'i_mid_{m}' for m in months]]
    monthly_data.columns = month_names
    
    # Melt data for visualization
    melted_data = monthly_data.melt(var_name='Month', value_name='Concentration')
    
    # Create enhanced box plot with violin overlay
    fig = px.violin(
        melted_data,
        x='Month',
        y='Concentration',
        box=True,
        title='🌊 Seasonal Microplastic Concentration Patterns',
        color='Month',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Add statistical annotations
    fig.update_layout(
        title=dict(x=0.5, font=dict(size=18)),
        xaxis=dict(
            tickangle=45,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            title='Concentration (particles/km²)',
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600,
        showlegend=False
    )
    
    return fig

def plot_severity_analysis(df):
    """Enhanced severity analysis with multi-dimensional insights."""
    # Create subplot with multiple perspectives
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Concentration vs Area', 'Severity Distribution', 
                       'Weight vs Runoff', 'Risk Assessment'),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Scatter plot 1: Concentration vs Area
    fig.add_trace(
        go.Scatter(
            x=df["area"],
            y=df["i_high"],
            mode='markers',
            marker=dict(
                size=df["mpw"]/df["mpw"].max()*20 + 5,
                color=df["runoff_jul"],
                colorscale='Viridis',
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            name='Area vs Concentration',
            hovertemplate='<b>Area:</b> %{x:.1f} km²<br><b>Concentration:</b> %{y:.1f}<br><extra></extra>'
        ),
        row=1, col=1
    )
    
    # Histogram: Severity distribution
    fig.add_trace(
        go.Histogram(
            x=df["i_high"],
            nbinsx=30,
            name='Severity Distribution',
            marker_color='rgba(255, 100, 100, 0.7)',
            marker_line_color='white',
            marker_line_width=1
        ),
        row=1, col=2
    )
    
    # Scatter plot 2: Weight vs Runoff
    fig.add_trace(
        go.Scatter(
            x=df["mpw"],
            y=df["runoff_jul"],
            mode='markers',
            marker=dict(
                size=8,
                color=df["i_mid"],
                colorscale='Plasma',
                opacity=0.7
            ),
            name='Weight vs Runoff'
        ),
        row=2, col=1
    )
    
    # Risk assessment bubble chart
    risk_score = (df["i_high"] * df["mpw"]) / df["area"]
    fig.add_trace(
        go.Scatter(
            x=df["i_high"],
            y=df["mpw"],
            mode='markers',
            marker=dict(
                size=risk_score/risk_score.max()*30 + 5,
                color=risk_score,
                colorscale='Reds',
                opacity=0.7,
                line=dict(width=1, color='darkred')
            ),
            name='Risk Assessment'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text='⚠️ Comprehensive Severity Analysis Dashboard',
        title_x=0.5,
        height=800,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Area (km²)", row=1, col=1)
    fig.update_yaxes(title_text="High Concentration", row=1, col=1)
    fig.update_xaxes(title_text="Concentration", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_xaxes(title_text="Microplastic Weight", row=2, col=1)
    fig.update_yaxes(title_text="July Runoff", row=2, col=1)
    fig.update_xaxes(title_text="High Concentration", row=2, col=2)
    fig.update_yaxes(title_text="Microplastic Weight", row=2, col=2)
    
    return fig

# Additional Enhancement Functions

def create_system_health_gauge(df):
    """Creates a system health gauge based on data quality."""
    # Calculate data quality metrics
    total_records = len(df)
    valid_coordinates = len(df.dropna(subset=['X', 'Y']))
    valid_concentrations = len(df.dropna(subset=['i_mid']))
    
    coordinate_quality = (valid_coordinates / total_records) * 100 if total_records > 0 else 0
    concentration_quality = (valid_concentrations / total_records) * 100 if total_records > 0 else 0
    overall_health = (coordinate_quality + concentration_quality) / 2
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = overall_health,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "📊 Data Quality Score", 'font': {'size': 20}},
        delta = {'reference': 90, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'lightgray'},
                {'range': [50, 80], 'color': 'yellow'},
                {'range': [80, 100], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    # Missing functions to add to utils2.py

import psutil
import time
import random
from datetime import datetime, timedelta

def create_animated_gauge(system_metrics):
    """Creates an animated system health gauge with real-time metrics."""
    fig = go.Figure()
    
    # CPU gauge
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = system_metrics.get('cpu_percent', 0),
        domain = {'row': 0, 'column': 0},
        title = {'text': "💻 CPU Usage (%)", 'font': {'size': 16}},
        delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    # Memory gauge
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = system_metrics.get('memory_percent', 0),
        domain = {'row': 0, 'column': 1},
        title = {'text': "💾 Memory Usage (%)", 'font': {'size': 16}},
        delta = {'reference': 60, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgreen"},
            'bar': {'color': "darkgreen"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'lightgreen'},
                {'range': [40, 80], 'color': 'yellow'},
                {'range': [80, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    # Temperature gauge (simulated)
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = system_metrics.get('cpu_temp', 45),
        domain = {'row': 1, 'column': 0},
        title = {'text': "🌡️ CPU Temp (°C)", 'font': {'size': 16}},
        delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkorange"},
            'bar': {'color': "darkorange"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'lightgreen'},
                {'range': [50, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    # GPU temperature gauge (simulated)
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = system_metrics.get('gpu_temp', 40),
        domain = {'row': 1, 'column': 1},
        title = {'text': "🎮 GPU Temp (°C)", 'font': {'size': 16}},
        delta = {'reference': 55, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "purple"},
            'bar': {'color': "purple"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 55], 'color': 'lightgreen'},
                {'range': [55, 75], 'color': 'yellow'},
                {'range': [75, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(
        grid = {'rows': 2, 'columns': 2, 'pattern': "independent"},
        height=600,
        title={
            'text': '⚡ System Health Dashboard',
            'x': 0.5,
            'font': {'size': 20, 'color': '#2E86C1'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_real_time_metrics_chart(system_history):
    """Creates a real-time metrics chart showing system performance over time."""
    if len(system_history) < 2:
        return go.Figure()
    
    # Convert system history to DataFrame for easier plotting
    df_history = pd.DataFrame(system_history)
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('📊 CPU & Memory Usage Over Time', '🌡️ Temperature Trends'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )
    
    # CPU usage line
    fig.add_trace(
        go.Scatter(
            x=df_history.index,
            y=df_history.get('cpu_percent', [0] * len(df_history)),
            mode='lines+markers',
            name='CPU Usage (%)',
            line=dict(color='#3498DB', width=3),
            marker=dict(size=6),
            hovertemplate='<b>CPU:</b> %{y:.1f}%<br><extra></extra>'
        ),
        row=1, col=1
    )
    
    # Memory usage line
    fig.add_trace(
        go.Scatter(
            x=df_history.index,
            y=df_history.get('memory_percent', [0] * len(df_history)),
            mode='lines+markers',
            name='Memory Usage (%)',
            line=dict(color='#E74C3C', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Memory:</b> %{y:.1f}%<br><extra></extra>'
        ),
        row=1, col=1
    )
    
    # CPU temperature
    fig.add_trace(
        go.Scatter(
            x=df_history.index,
            y=df_history.get('cpu_temp', [45] * len(df_history)),
            mode='lines+markers',
            name='CPU Temp (°C)',
            line=dict(color='#F39C12', width=3),
            marker=dict(size=6),
            hovertemplate='<b>CPU Temp:</b> %{y:.1f}°C<br><extra></extra>'
        ),
        row=2, col=1
    )
    
    # GPU temperature
    fig.add_trace(
        go.Scatter(
            x=df_history.index,
            y=df_history.get('gpu_temp', [40] * len(df_history)),
            mode='lines+markers',
            name='GPU Temp (°C)',
            line=dict(color='#9B59B6', width=3),
            marker=dict(size=6),
            hovertemplate='<b>GPU Temp:</b> %{y:.1f}°C<br><extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add warning zones
    fig.add_hline(y=70, line_dash="dash", line_color="orange", 
                  annotation_text="High Usage Warning", row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                  annotation_text="High Temperature Warning", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=700,
        title={
            'text': '📈 Real-Time System Performance Monitor',
            'x': 0.5,
            'font': {'size': 18, 'color': '#2E86C1'}
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x-axes
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.2)')
    
    # Update y-axes
    fig.update_yaxes(title_text="Usage (%)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.2)')
    
    return fig

def create_interactive_3d_scatter(df, sample_size=1500):
    """Creates an enhanced interactive 3D scatter plot with clustering and animations."""
    if df.empty or len(df) == 0:
        st.warning("DataFrame is empty. Cannot create 3D visualization.")
        return go.Figure()
    
    # Sample data for performance
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        st.info(f"📊 Displaying sample of {sample_size:,} points from {len(df):,} total records")
    else:
        df_sample = df.copy()
    
    # Clean data
    required_cols = ['X', 'Y', 'i_mid']
    if not all(col in df_sample.columns for col in required_cols):
        st.error(f"Missing required columns: {required_cols}")
        return go.Figure()
    
    df_clean = df_sample.dropna(subset=required_cols)
    if df_clean.empty:
        st.warning("No valid data points after cleaning.")
        return go.Figure()
    
    # Calculate additional metrics for enhanced visualization
    df_clean = df_clean.copy()
    df_clean['log_concentration'] = np.log1p(df_clean['i_mid'])
    df_clean['size_factor'] = np.clip(df_clean['i_mid'] / df_clean['i_mid'].quantile(0.95) * 15 + 3, 3, 20)
    
    # Create color categories based on concentration levels
    concentration_quantiles = df_clean['i_mid'].quantile([0.25, 0.5, 0.75, 0.9])
    def categorize_concentration(value):
        if value <= concentration_quantiles[0.25]:
            return 'Low'
        elif value <= concentration_quantiles[0.5]:
            return 'Moderate'
        elif value <= concentration_quantiles[0.75]:
            return 'High'
        elif value <= concentration_quantiles[0.9]:
            return 'Very High'
        else:
            return 'Critical'
    
    df_clean['concentration_category'] = df_clean['i_mid'].apply(categorize_concentration)
    
    # Create the 3D scatter plot
    fig = go.Figure()
    
    # Color scheme for categories
    category_colors = {
        'Low': '#2ECC71',
        'Moderate': '#F1C40F', 
        'High': '#E67E22',
        'Very High': '#E74C3C',
        'Critical': '#8E44AD'
    }
    
    # Add traces for each category
    for category in df_clean['concentration_category'].unique():
        category_data = df_clean[df_clean['concentration_category'] == category]
        
        fig.add_trace(go.Scatter3d(
            x=category_data['X'],
            y=category_data['Y'],
            z=category_data['i_mid'],
            mode='markers',
            marker=dict(
                size=category_data['size_factor'],
                color=category_colors.get(category, '#3498DB'),
                opacity=0.8,
                line=dict(width=0.5, color='white'),
                colorscale=None
            ),
            name=f'{category} ({len(category_data)} points)',
            text=[
                f'<b>Location:</b> {x:.2f}°E, {y:.2f}°N<br>'
                f'<b>Concentration:</b> {c:.2f} particles/km²<br>'
                f'<b>Category:</b> {cat}<br>'
                f'<b>Log Concentration:</b> {log_c:.2f}'
                for x, y, c, cat, log_c in zip(
                    category_data['X'], category_data['Y'], 
                    category_data['i_mid'], category_data['concentration_category'],
                    category_data['log_concentration']
                )
            ],
            hovertemplate='%{text}<extra></extra>',
            showlegend=True
        ))
    
    # Add statistical surfaces (optional - can be toggled)
    if len(df_clean) > 100:
        # Create a surface showing concentration trends
        try:
            from scipy.interpolate import griddata
            
            # Create grid for surface
            xi = np.linspace(df_clean['X'].min(), df_clean['X'].max(), 20)
            yi = np.linspace(df_clean['Y'].min(), df_clean['Y'].max(), 20)
            Xi, Yi = np.meshgrid(xi, yi)
            
            # Interpolate concentration values
            Zi = griddata(
                (df_clean['X'], df_clean['Y']), 
                df_clean['log_concentration'],
                (Xi, Yi), 
                method='cubic', 
                fill_value=0
            )
            
            # Add surface
            fig.add_trace(go.Surface(
                x=xi,
                y=yi,
                z=Zi,
                opacity=0.3,
                colorscale='Viridis',
                showscale=False,
                name='Concentration Surface',
                hoverinfo='skip'
            ))
        except ImportError:
            pass  # Skip surface if scipy not available
    
    # Enhanced layout with professional styling
    fig.update_layout(
        title=dict(
            text='🌍 Interactive 3D Microplastic Distribution Analysis',
            x=0.5,
            font=dict(size=20, color='#2E86C1', family='Arial Black')
        ),
        scene=dict(
            xaxis=dict(
                title='Longitude (°E)',
                titlefont=dict(size=14, color='#34495E'),
                tickfont=dict(size=12),
                gridcolor='rgba(255,255,255,0.3)',
                showbackground=True,
                backgroundcolor='rgba(230,230,250,0.1)',
                showspikes=False
            ),
            yaxis=dict(
                title='Latitude (°N)',
                titlefont=dict(size=14, color='#34495E'),
                tickfont=dict(size=12),
                gridcolor='rgba(255,255,255,0.3)',
                showbackground=True,
                backgroundcolor='rgba(230,250,230,0.1)',
                showspikes=False
            ),
            zaxis=dict(
                title='Concentration (particles/km²)',
                titlefont=dict(size=14, color='#34495E'),
                tickfont=dict(size=12),
                gridcolor='rgba(255,255,255,0.3)',
                showbackground=True,
                backgroundcolor='rgba(250,230,230,0.1)',
                showspikes=False,
                type='log' if df_clean['i_mid'].max() / df_clean['i_mid'].min() > 100 else 'linear'
            ),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=800,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        )
    )
    
    # Add custom annotations
    max_point = df_clean.loc[df_clean['i_mid'].idxmax()]
    fig.add_trace(go.Scatter3d(
        x=[max_point['X']],
        y=[max_point['Y']],
        z=[max_point['i_mid']],
        mode='markers+text',
        marker=dict(
            size=20,
            color='red',
            symbol='diamond',
            line=dict(width=3, color='white')
        ),
        text=['📍 MAX'],
        textposition='top center',
        textfont=dict(size=12, color='red', family='Arial Black'),
        name='Highest Concentration',
        hovertemplate=f'<b>MAXIMUM POINT</b><br>Location: {max_point["X"]:.2f}°E, {max_point["Y"]:.2f}°N<br>Concentration: {max_point["i_mid"]:.2f}<extra></extra>',
        showlegend=False
    ))
    
    return fig

def get_system_metrics():
    """Gets real-time system metrics for monitoring."""
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Simulate temperature readings (replace with actual sensor readings if available)
        # In a real implementation, you'd use libraries like:
        # - psutil for some temperature readings
        # - specific hardware libraries for GPU temps
        cpu_temp = random.uniform(35, 75)  # Simulated CPU temperature
        gpu_temp = random.uniform(30, 80)  # Simulated GPU temperature
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Get network I/O
        net_io = psutil.net_io_counters()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'cpu_temp': cpu_temp,
            'gpu_temp': gpu_temp,
            'disk_percent': disk_percent,
            'network_sent': net_io.bytes_sent,
            'network_recv': net_io.bytes_recv,
            'timestamp': datetime.now()
        }
    except Exception as e:
        st.warning(f"Could not retrieve system metrics: {e}")
        return {
            'cpu_percent': random.uniform(10, 60),
            'memory_percent': random.uniform(20, 80),
            'cpu_temp': random.uniform(35, 65),
            'gpu_temp': random.uniform(30, 70),
            'disk_percent': random.uniform(20, 90),
            'network_sent': 0,
            'network_recv': 0,
            'timestamp': datetime.now()
        }
    fig.update_layout(height=400, font={'color': "darkblue", 'family': "Arial"})
    return fig

