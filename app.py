import streamlit as st
import plotly.graph_objs as go
import pandas as pd
from utils2 import (
    load_data,
    get_variable_explanations,
    spatial_analysis_colab,
    plot_total_monthly_input,
    plot_top_rivers,
    plot_plastic_heatmap_folium,
    plot_top_rivers_by_month,
    plot_correlational_analysis,
    plot_hotspot_analysis,
    plot_seasonal_analysis,
    plot_severity_analysis,
    load_hotspot_data,
    create_animated_gauge,
    create_real_time_metrics_chart,
    # create_interactive_3d_scatter
)
from streamlit_folium import folium_static
from pyspark.sql import SparkSession
import time

# Import necessary modules for system monitoring
import psutil
import numpy as np
from datetime import datetime
import plotly.express as px

# Create a Spark session
spark = SparkSession.builder \
    .appName("Microplastic Analysis") \
    .getOrCreate()

# --- System Monitoring Functions ---
def get_wmi_temperatures():
    """Fallback WMI temperature monitoring for Windows"""
    try:
        import wmi
        w = wmi.WMI(namespace=r"root\wmi")
        temperature_info = w.MSAcpi_ThermalZoneTemperature()
        if temperature_info:
            cpu_temp = (temperature_info[0].CurrentTemperature / 10.0) - 273.15
            return cpu_temp, cpu_temp
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
    cpu_percent = psutil.cpu_percent(interval=0.1)
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

# --- Page Configuration ---
st.set_page_config(
    page_title="🌊 Microplastic Analytics Hub",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
    }
    
    .main > div {
        padding-top: 2rem;
    }
    
    /* Glassmorphism effect for containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    /* Animated title */
    .main-title {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 3s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem !important;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05));
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .metric-title {
        font-size: 0.9rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px);
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(5px);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 50px;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* Pulse animation for warning states */
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Status indicators */
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-good { background-color: #4ecdc4; }
    .status-warning { background-color: #ffe66d; }
    .status-critical { background-color: #ff6b6b; }
    
    /* Loading animation */
    .loading-spinner {
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top: 3px solid #667eea;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        display: inline-block;
        margin-right: 10px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Navigation tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1.5rem;
        color: rgba(255, 255, 255, 0.8);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system_history' not in st.session_state:
    st.session_state.system_history = []
if 'show_data_view' not in st.session_state:
    st.session_state.show_data_view = False
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# --- Load Data (cached) ---
@st.cache_data
def get_data():
    return load_data('D:\Documents\BDA_EL\water_pollution.csv')

@st.cache_data
def get_hotspot_data():
    return load_hotspot_data("results/hotspot_analysis.txt")

# Load data
with st.spinner("🔄 Loading data..."):
    df = get_data()
    hotspot_df = get_hotspot_data()

# --- Enhanced Sidebar ---
with st.sidebar:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("## 🖥️ System Monitor")
    
    # Auto-refresh toggle
    auto_refresh = st.toggle("🔄 Auto Refresh", value=st.session_state.auto_refresh)
    if auto_refresh != st.session_state.auto_refresh:
        st.session_state.auto_refresh = auto_refresh
    
    # Real-time system metrics
    system_metrics = get_system_metrics()
    
    # Store metrics history for trending
    st.session_state.system_history.append(system_metrics)
    if len(st.session_state.system_history) > 50:  # Keep last 50 readings
        st.session_state.system_history.pop(0)
    
    # Status determination
    cpu_status = "good" if system_metrics['cpu_percent'] < 70 else ("warning" if system_metrics['cpu_percent'] < 90 else "critical")
    mem_status = "good" if system_metrics['memory_percent'] < 70 else ("warning" if system_metrics['memory_percent'] < 90 else "critical")
    temp_status = "good" if system_metrics['cpu_temp'] < 70 else ("warning" if system_metrics['cpu_temp'] < 85 else "critical")
    
    # Enhanced metric cards
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">
            <span class="status-indicator status-{cpu_status}"></span>CPU Usage
        </div>
        <div class="metric-value">{system_metrics['cpu_percent']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">
            <span class="status-indicator status-{mem_status}"></span>Memory Usage
        </div>
        <div class="metric-value">{system_metrics['memory_percent']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">
            <span class="status-indicator status-{temp_status}"></span>CPU Temperature
        </div>
        <div class="metric-value">{system_metrics['cpu_temp']:.1f}°C</div>
    </div>
    """, unsafe_allow_html=True)
    
    # System health gauge
    if len(st.session_state.system_history) > 5:
        gauge_fig = create_animated_gauge(system_metrics)
        st.plotly_chart(gauge_fig, use_container_width=True, key="gauge_chart")
    
    # Temperature warning with animation
    if system_metrics['cpu_temp'] > 70 or system_metrics['gpu_temp'] > 75:
        st.markdown('<div class="pulse">', unsafe_allow_html=True)
        st.error("🔥 High system temperature detected!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Real-time metrics chart
    if len(st.session_state.system_history) > 10:
        metrics_fig = create_real_time_metrics_chart(st.session_state.system_history)
        st.plotly_chart(metrics_fig, use_container_width=True, key="metrics_chart")
    
    st.markdown("---")
    
    # Enhanced navigation
    st.markdown("### 📊 Dashboard Navigation")
    
    if st.button("📈 Analytics Dashboard" if st.session_state.show_data_view else "📋 Data Overview", 
                 use_container_width=True):
        st.session_state.show_data_view = not st.session_state.show_data_view
        st.rerun()
    
    # Quick stats
    st.markdown("### 📈 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Variables", df.shape[1])
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main Content ---
st.markdown('<h1 class="main-title">🌊 Microplastic Analytics Hub</h1>', unsafe_allow_html=True)

# Auto-refresh mechanism
if st.session_state.auto_refresh:
    placeholder = st.empty()
    with placeholder:
        st.info("🔄 Auto-refresh enabled - Dashboard updates every 5 seconds")
    time.sleep(0.1)  # Brief display
    placeholder.empty()

if st.session_state.show_data_view:
    # Enhanced Data Overview
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📋 Data Overview</h2>', unsafe_allow_html=True)
    
    # Tabs for different data views
    tab1, tab2, tab3 = st.tabs(["📊 Dataset Preview", "📈 Statistics", "ℹ️ Variable Info"])
    
    with tab1:
        st.dataframe(
            df.head(20), 
            use_container_width=True,
            height=400
        )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024**2:.1f} MB")
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
    
    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Data quality visualization
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            fig_missing = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                title="Missing Values by Column",
                labels={'y': 'Missing Count', 'x': 'Columns'}
            )
            fig_missing.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_missing, use_container_width=True)
    
    with tab3:
        st.markdown(get_variable_explanations())
    
    st.markdown('</div>', unsafe_allow_html=True)
    
else:
    # # Enhanced Analytics Dashboard
    # st.markdown('<h2 class="section-header">🔬 Advanced Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # # Interactive 3D scatter plot as hero visualization
    # st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    # st.subheader("🌍 3D Interactive Microplastic Distribution")
    # fig_3d = create_interactive_3d_scatter(df)
    # st.plotly_chart(fig_3d, use_container_width=True, key="3d_scatter")
    # st.markdown('</div>', unsafe_allow_html=True)
    
    # # Two-column layout for visualizations
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     # Graph 1: Enhanced Spatial Distribution
    #     st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    #     st.subheader("📍 Spatial Distribution Analysis")
    #     with st.spinner("Generating spatial analysis..."):
    #         spatial_sample = spatial_analysis_colab(df, sample_size=2000)
    #     with st.expander("💡 Analysis Insights", expanded=False):
    #         st.markdown("""
    #         **Key Insights:** This visualization reveals geographic hotspots of microplastic contamination.
    #         Higher concentrations (warmer colors) indicate areas requiring immediate attention.
            
    #         **Variables:**
    #         - **X, Y:** Geographic coordinates (Longitude, Latitude)
    #         - **i_mid:** Microplastic concentration intensity
    #         """)
    #     st.markdown('</div>', unsafe_allow_html=True)
    
    # with col2:
    #     # Graph 2: Enhanced Monthly Analysis
    #     st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    #     st.subheader("📅 Monthly Pollution Trends")
    #     with st.spinner("Analyzing monthly trends..."):
    #         monthly_totals = plot_total_monthly_input(df)
    #     with st.expander("💡 Seasonal Insights", expanded=False):
    #         st.markdown("""
    #         **Trend Analysis:** Seasonal patterns reveal pollution peaks during specific months,
    #         often correlating with industrial activity and weather patterns.
            
    #         **Variables:**
    #         - **Monthly mid-estimates:** Aggregated pollution data per month
    #         """)
    #     st.markdown('</div>', unsafe_allow_html=True)
    
    # Full-width advanced visualizations
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.subheader("🗺️ Geographic Pollution Hotspots")
    
    # Tabs for different map views
    map_tab1, map_tab2 = st.tabs(["🔥 Top Polluting Rivers", "🌡️ Heat Distribution"])
    
    with map_tab1:
        with st.spinner("Loading river pollution data..."):
            folium_map = plot_top_rivers(df)
        if folium_map:
            folium_static(folium_map, width=700, height=500)
        else:
            st.warning("Could not generate river pollution map.")
    
    with map_tab2:
        with st.spinner("Generating heat map..."):
            heatmap = plot_plastic_heatmap_folium(df)
        if heatmap:
            folium_static(heatmap, width=700, height=500)
        else:
            st.warning("Failed to generate heatmap.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced Analytics Row
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.subheader("📊 Advanced Analytics Suite")
    
    analysis_tabs = st.tabs([
        "🎬 Animated Rivers", "🔗 Correlations", "🎯 Hotspots", 
        "🌦️ Seasonal Patterns", "⚠️ Severity Analysis"
    ])
    
    with analysis_tabs[0]:
        st.markdown("### Animated River Pollution by Month")
        with st.spinner("Creating animation..."):
            fig5 = plot_top_rivers_by_month(df)
        st.plotly_chart(fig5, use_container_width=True, key="animated_rivers")
    
    with analysis_tabs[1]:
        st.markdown("### Correlation Matrix Analysis")
        with st.spinner("Computing correlations..."):
            fig6 = plot_correlational_analysis(df)
        st.plotly_chart(fig6, use_container_width=True, key="correlations")
    
    with analysis_tabs[2]:
        st.markdown("### Statistical Hotspot Detection")
        with st.spinner("Analyzing hotspots..."):
            fig7 = plot_hotspot_analysis(hotspot_df)
        st.plotly_chart(fig7, use_container_width=True, key="hotspots")
    
    with analysis_tabs[3]:
        st.markdown("### Seasonal Variation Analysis")
        with st.spinner("Analyzing seasonal patterns..."):
            fig8 = plot_seasonal_analysis(df)
        st.plotly_chart(fig8, use_container_width=True, key="seasonal")
    
    with analysis_tabs[4]:
        st.markdown("### Environmental Impact Severity")
        with st.spinner("Computing severity metrics..."):
            fig9 = plot_severity_analysis(df)
        st.plotly_chart(fig9, use_container_width=True, key="severity")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; padding: 2rem;">
    <p>🌊 Microplastic Analytics Hub | Powered by Advanced Data Science</p>
    <p>Real-time monitoring • Interactive visualizations • Predictive insights</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh logic
if st.session_state.auto_refresh:
    time.sleep(5)
    st.rerun()