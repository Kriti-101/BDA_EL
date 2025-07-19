import streamlit as st
import plotly.graph_objs as go
import pandas as pd
from utils import (
    load_data,
    get_variable_explanations,
    spatial_analysis_colab,
    plot_total_monthly_input,
    plot_top_rivers,
    plot_plastic_heatmap_folium,
    plot_top_rivers_by_month,
    plot_correlational_analysis,
    plot_hotspot_analysis, # This will now use the new plot
    plot_seasonal_analysis,
    plot_severity_analysis,
    load_hotspot_data # <--- IMPORT THE NEW FUNCTION
)
from streamlit_folium import folium_static
from pyspark.sql import SparkSession

# Import necessary modules for system monitoring
import psutil
import numpy as np # For temperature simulation fallback
from datetime import datetime

# Create a Spark session
spark = SparkSession.builder \
    .appName("Microplastic Analysis") \
    .getOrCreate()

# --- System Monitoring Functions (Copied from microplastic_dashboard.py) ---
def get_wmi_temperatures():
    """Fallback WMI temperature monitoring for Windows"""
    try:
        import wmi
        w = wmi.WMI(namespace=r"root\wmi")
        temperature_info = w.MSAcpi_ThermalZoneTemperature()
        if temperature_info:
            cpu_temp = (temperature_info[0].CurrentTemperature / 10.0) - 273.15
            return cpu_temp, cpu_temp  # Use CPU temp for both if GPU not available
    except:
        pass
    return None, None

def get_system_temperatures():
    """Get system temperatures from multiple sources"""
    cpu_temp = None
    gpu_temp = None

    # Try psutil first (most reliable)
    if hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures()

            # Try different sensor names for CPU
            if "coretemp" in temps:
                for sensor in temps["coretemp"]:
                    if "Package id 0" in sensor.label:
                        cpu_temp = sensor.current
                        break
                    elif "Core 0" in sensor.label and cpu_temp is None:
                        cpu_temp = sensor.current

            # Try different sensor names for GPU
            if "nvidia" in temps:
                gpu_temp = temps["nvidia"][0].current
            elif "amdgpu" in temps:
                gpu_temp = temps["amdgpu"][0].current

        except Exception as e:
            st.sidebar.warning(f"psutil temperature error: {e}")

    # Try WMI as fallback (Windows only)
    if cpu_temp is None or gpu_temp is None:
        try:
            wmi_cpu, wmi_gpu = get_wmi_temperatures()
            if cpu_temp is None:
                cpu_temp = wmi_cpu
            if gpu_temp is None:
                gpu_temp = wmi_gpu
        except Exception as e:
            st.sidebar.warning(f"WMI temperature error: {e}")

    # Fallback to simulation if no real data
    if cpu_temp is None:
        cpu_temp = np.random.uniform(45, 75)
    if gpu_temp is None:
        gpu_temp = np.random.uniform(40, 70)

    return float(cpu_temp), float(gpu_temp)

def get_system_metrics():
    """Get comprehensive system metrics"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    cpu_temp, gpu_temp = get_system_temperatures()

    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_used_gb': memory.used / (1024**3),
        'memory_total_gb': memory.total / (1024**3),
        'disk_percent': disk.percent,
        'cpu_temp': cpu_temp,
        'gpu_temp': gpu_temp,
        'timestamp': datetime.now()
    }
# --- End System Monitoring Functions ---


# --- Page Configuration ---
st.set_page_config(
    page_title="Microplastic Distribution Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Aesthetics ---
st.markdown("""
<style>
    /* General styling for the app */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    /* Adjust sidebar width if needed */
    section.main .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* Style for headings */
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary-color);
    }
    /* Style for expander (dropdown) */
    .streamlit-expanderHeader {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
    }
    .streamlit-expanderContent {
        background-color: var(--secondary-background-color);
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    /* Dataframe styling */
    .stDataFrame {
        color: var(--text-color);
        background-color: var(--secondary-background-color);
    }
    /* Buttons */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #1a6bb2;
    }
    /* Custom sidebar metric styling */
    .sidebar-metric-container {
        background-color: #e0f2f7; /* Light blue background */
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        border: 1px solid #b3e5fc; /* Lighter blue border */
    }
    .sidebar-metric-title {
        font-size: 0.9rem;
        color: #01579b; /* Darker blue text */
        font-weight: bold;
        margin-bottom: 5px;
    }
    .sidebar-metric-value {
        font-size: 1.1rem;
        color: #007bb5; /* Blue value text */
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data (once) ---
@st.cache_data
def get_data():
    return load_data('D:\Documents\BDA_EL\water_pollution.csv')

# Load hotspot data once and cache it
@st.cache_data
def get_hotspot_data():
    return load_hotspot_data("hotspot_analysis.txt") # Ensure this path is correct

df = get_data()
hotspot_df = get_hotspot_data() # Load hotspot data here

# --- Sidebar (Left Panel) ---
with st.sidebar:
    st.header("🖥️ System Monitor") # Changed header to "System Monitor"

    # System Monitoring content (always visible)
    system_metrics = get_system_metrics()

    st.markdown("### Current System Status")

    # Using custom styling for metrics
    st.markdown(f"""
    <div class="sidebar-metric-container">
        <div class="sidebar-metric-title">CPU Usage</div>
        <div class="sidebar-metric-value">{system_metrics['cpu_percent']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-metric-container">
        <div class="sidebar-metric-title">Memory Usage</div>
        <div class="sidebar-metric-value">{system_metrics['memory_percent']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-metric-container">
        <div class="sidebar-metric-title">CPU Temperature</div>
        <div class="sidebar-metric-value">{system_metrics['cpu_temp']:.1f}°C</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-metric-container">
        <div class="sidebar-metric-title">GPU Temperature</div>
        <div class="sidebar-metric-value">{system_metrics['gpu_temp']:.1f}°C</div>
    </div>
    """, unsafe_allow_html=True)

    # Temperature warning
    if system_metrics['cpu_temp'] > 70 or system_metrics['gpu_temp'] > 75:
        st.warning("⚠️ High system temperature detected!")

    st.markdown("---") # Separator

    # Button to toggle between Data and Graphs in the main content
    if 'show_data_view' not in st.session_state:
        st.session_state.show_data_view = False # False means show graphs by default

    if st.button("Show Data Overview" if not st.session_state.show_data_view else "Show Graphs"):
        st.session_state.show_data_view = not st.session_state.show_data_view

# --- Main Content (Right Panel) ---
st.title("Data driven analysis of micro plastic distribution")
st.markdown("---")

if st.session_state.show_data_view:
    # Display Data Overview
    st.subheader("water_pollution.csv Data Overview")
    st.dataframe(df.head())
    st.write(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
    st.markdown(get_variable_explanations())
else:
    # Display Graphs
    # Graph 1: Spatial distribution of micro plastic concentration
    st.subheader("1. Spatial Distribution of Microplastic Concentration")
    spatial_sample = spatial_analysis_colab(df, sample_size=2000)
    with st.expander("Inference & Variables for Spatial Distribution"):
        st.markdown("""
        **Inference:** This scatter plot visualizes the geographical spread of microplastic concentration (i_mid) across different locations. The color intensity reflects the concentration.
        **Variables Used:**
        *   **X, Y:** Longitude and Latitude for plotting locations.
        *   **i_mid:** Mid-estimate of microplastic concentration, used for color.
        """)

    # Graph 2: Total monthly plastic input (mid estimate)
    st.subheader("2. Total Monthly Plastic Input (Mid Estimate)")
    monthly_totals = plot_total_monthly_input(df)
    with st.expander("Inference & Variables for Total Monthly Plastic Input"):
        st.markdown("""
        **Inference:** This bar chart shows the aggregated mid-estimate of microplastic input for each month across all locations. It helps in understanding seasonal trends in plastic pollution, potentially correlating with rainfall patterns or human activities.
        **Variables Used:**
        *   **i_mid_jan, i_mid_feb, ..., i_mid_dec:** Mid-estimates of microplastic concentration for each respective month.
        """)

    # Graph 3: Top twenty polluting rivers
    st.subheader("3. Top Twenty Polluting Rivers")
    folium_map = plot_top_rivers(df)
    if folium_map:
        folium_static(folium_map, width=700, height=500)
    else:
        st.warning("Could not generate or display the Top Rivers map.")
    with st.expander("Inference & Variables for Top Twenty Polluting Rivers"):
        st.markdown("""
        **Inference:** This map identifies and ranks rivers based on their contribution to microplastic pollution, highlighting major sources.
        **Variables Used:**
        *   **X, Y:** Geographic coordinates.
        *   **i_mid:** Mid-estimate of microplastic concentration.
        """)

    # Graph 4: Heatmap of plastic (now Folium-based)
    st.subheader("4. Heatmap of Plastic Concentration")
    heatmap = plot_plastic_heatmap_folium(df)
    if heatmap:
        folium_static(heatmap, width=700, height=500)
    else:
        st.warning("Failed to generate heatmap. Check data requirements.")
    with st.expander("Inference & Variables for Heatmap of Plastic"):
        st.markdown("""
        **Inference:** This heatmap shows global microplastic concentration hotspots.
        Red indicates the highest pollution levels, while blue shows lower concentrations.
        **Variables Used:**
        *   **X, Y:** Geographic coordinates.
        *   **i_mid:** Mid-estimate of microplastic concentration (weight).
        """)

    # Graph 5: Top river plastic input by month (now animated Plotly bar)
    st.subheader("5. Top River Plastic Input by Month")
    fig5 = plot_top_rivers_by_month(df)
    st.plotly_chart(fig5, use_container_width=True)
    with st.expander("Inference & Variables for Top River Plastic Input by Month"):
        st.markdown("""
        **Inference:** This animated bar chart shows the monthly plastic input for the top 20 rivers, allowing for a detailed seasonal analysis of their contribution.
        **Variables Used:**
        *   **monthly_input:** Monthly input of plastic (tonnes).
        *   **river_id:** Unique identifier for each river based on coordinates (X_Y).
        *   **mpw:** Microplastic weight (kg).
        """)

    # Graph 6: Correlational Analysis
    st.subheader("6. Correlational Analysis")
    fig6 = plot_correlational_analysis(df)
    st.plotly_chart(fig6, use_container_width=True)
    with st.expander("Inference & Variables for Correlational Analysis"):
        st.markdown("""
        **Inference:** This heatmap displays the correlation matrix between selected variables, helping identify relationships between microplastic concentration and factors like runoff, total plastic weight, and area.
        **Variables Used:**
        *   **i_mid:** Mid-estimate of microplastic concentration.
        *   **runoff_jan (and other runoff months):** Monthly runoff data.
        *   **mpw:** Microplastic weight.
        *   **area:** Area of the sampling location.
        """)

    # Graph 7: Hotspot Analysis (UPDATED)
    st.subheader("7. Hotspot Analysis")
    # Pass the pre-loaded hotspot_df to the plotting function
    fig7 = plot_hotspot_analysis(hotspot_df)
    st.plotly_chart(fig7, use_container_width=True)
    with st.expander("Inference & Variables for Hotspot Analysis"):
        st.markdown("""
        **Inference:** This map visualizes hotspot locations based on a calculated hotspot score. Larger, more intensely colored circles indicate areas with higher hotspot scores, suggesting statistically significant clusters of high microplastic concentration.
        **Variables Used:**
        *   **latitude, longitude:** Geographic coordinates of the sample locations.
        *   **hotspot_score:** The calculated score indicating the intensity of the hotspot.
        *   **grid:** Identifier for the grid cell.
        """)

    # Graph 8: Seasonal Analysis
    st.subheader("8. Seasonal Analysis")
    fig8 = plot_seasonal_analysis(df)
    st.plotly_chart(fig8, use_container_width=True)
    with st.expander("Inference & Variables for Seasonal Analysis"):
        st.markdown("""
        **Inference:** This box plot illustrates the distribution of microplastic concentrations across different months, revealing seasonal patterns.
        **Variables Used:**
        *   **i_mid_jan, ..., i_mid_dec:** Monthly mid-estimates of microplastic concentration.
        """)

    # Graph 9: Severity Analysis
    st.subheader("9. Severity Analysis")
    fig9 = plot_severity_analysis(df)
    st.plotly_chart(fig9, use_container_width=True)
    with st.expander("Inference & Variables for Severity Analysis"):
        st.markdown("""
        **Inference:** Severity analysis assesses the potential impact of microplastic pollution, exploring relationships between concentration, area, and runoff.
        **Variables Used:**
        *   **i_high:** High-estimate of microplastic concentration.
        *   **area:** The size of the area affected.
        *   **runoff_jul:** Runoff data for a specific month.
        *   **mpw:** Microplastic weight.
        """)

st.markdown("---")
st.markdown("BDT")
