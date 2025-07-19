import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv('D:\Documents\BDA_EL\water_pollution.csv')

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
          'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

# --- 1. Average Monthly Runoff Line Plot ---
monthly_avg = {month: df[f'runoff_{month}'].mean() for month in months}
monthly_avg_df = pd.DataFrame({
    'Month': [m.capitalize() for m in months],
    'Average Runoff': list(monthly_avg.values())
})

fig1 = px.line(
    monthly_avg_df, x='Month', y='Average Runoff', markers=True,
    title='Average Runoff per Month'
)
fig1.show()


# --- 2. Animated Scatter Plot: Area vs MPW Colored by i_mid ---
# Option 2: Drop rows where i_mid is NaN
df_clean = df.dropna(subset=['i_mid'])

fig2 = px.scatter(
    df_clean, x='area', y='mpw', color='i_mid',
    size='i_mid', hover_data=['area', 'mpw', 'i_mid'],
    title='Area vs MPW colored by i_mid',
    opacity=0.6
)
fig2.show()



# --- 3. Animated Map of Monthly Runoff ---
# Melt dataframe for easier animation
runoff_cols = [f'runoff_{month}' for month in months]
df_melt = df.melt(id_vars=['X', 'Y'], value_vars=runoff_cols,
                  var_name='Month', value_name='Runoff')
df_melt['Month'] = df_melt['Month'].str.replace('runoff_', '').str.capitalize()

# Prepare the melted dataframe again
runoff_cols = [f'runoff_{month}' for month in months]
df_melt = df.melt(id_vars=['X', 'Y'], value_vars=runoff_cols,
                  var_name='Month', value_name='Runoff')

df_melt['Month'] = df_melt['Month'].str.replace('runoff_', '').str.capitalize()

# Drop NaN Runoff values
df_melt_clean = df_melt.dropna(subset=['Runoff'])

fig3 = px.scatter_geo(
    df_melt_clean,
    lat='Y',
    lon='X',
    color='Runoff',
    size='Runoff',
    animation_frame='Month',
    projection='natural earth',
    title='Animated Runoff Map Across Months',
    color_continuous_scale='Viridis'
)
fig3.show()

