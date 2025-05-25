import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static

# Load data
df = pd.read_csv("spacex_final.csv")

# Display available columns for debugging
st.write("🧾 Columns in CSV:", df.columns.tolist())

# Extract year from 'Date' column if available
date_column = next((col for col in df.columns if 'date' in col.lower()), None)
if date_column:
    df['Year'] = pd.to_datetime(df[date_column]).dt.year
else:
    df['Year'] = 0  # fallback if date column missing

# Detect site column safely
site_column = next((col for col in df.columns if 'site' in col.lower()), None)

# Sidebar - Filter
st.sidebar.header("🔍 Filter Launch Data")
year_filter = st.sidebar.selectbox("Select Launch Year:", options=sorted(df['Year'].unique()), index=0)
if site_column:
    site_options = ['All'] + sorted(df[site_column].dropna().unique())
    site_filter = st.sidebar.selectbox("Select Launch Site:", options=site_options, index=0)
else:
    site_filter = 'All'
    st.warning("🚧 No column containing 'site' found. Skipping site filter.")

# Filter data
filtered_df = df[df['Year'] == year_filter]
if site_column and site_filter != 'All':
    filtered_df = filtered_df[filtered_df[site_column] == site_filter]

# Title
st.title("🚀 SpaceX Launch Analysis & Prediction Platform")
st.markdown("Explore historical launches and predict success using machine learning.")

# Show data
with st.expander("📄 View Filtered Data"):
    st.dataframe(filtered_df)

# Success by Orbit
st.subheader("📊 Success Rate by Orbit")
if not filtered_df.empty and 'Orbit' in filtered_df.columns:
    orbit_success = filtered_df.groupby("Orbit")['Class'].mean().sort_values(ascending=False)
    st.bar_chart(orbit_success)
else:
    st.info("No data or 'Orbit' column not found.")

# Map
st.subheader("🗺️ Launch Sites Map")
if 'Latitude' in df.columns and 'Longitude' in df.columns:
    map_df = filtered_df[['Latitude', 'Longitude', 'Class']].copy()
    if site_column and site_column in filtered_df.columns:
        map_df[site_column] = filtered_df[site_column]
    map_df = map_df.dropna(subset=['Latitude', 'Longitude'])
    if not map_df.empty:
        launch_map = folium.Map(location=[map_df['Latitude'].mean(), map_df['Longitude'].mean()], zoom_start=3)
        for _, row in map_df.iterrows():
            color = "green" if row['Class'] == 1 else "red"
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=row.get(site_column, "Launch Site") if site_column else "Launch Site",
                icon=folium.Icon(color=color)
            ).add_to(launch_map)
        folium_static(launch_map)
    else:
        st.warning("No valid coordinates for mapping.")
else:
    st.warning("Map data missing 'Latitude' or 'Longitude' columns.")

# ML Section
st.subheader("🤖 Predict Launch Success")
payload = st.slider("Payload Mass (kg):", int(df['Payload Mass (kg)'].min()), int(df['Payload Mass (kg)'].max()))

orbit = st.selectbox("Orbit:", sorted(df['Orbit'].dropna().unique()) if 'Orbit' in df.columns else [])
weather = st.selectbox("Weather:", sorted(df['Weather'].dropna().unique()) if 'Weather' in df.columns else [])

# One-hot encode
df_encoded = pd.get_dummies(df, columns=['Orbit', 'Weather'], drop_first=True)

cols_to_drop = ['Class', 'Year']
if date_column: cols_to_drop.append(date_column)
if site_column: cols_to_drop.append(site_column)
for col in ['Latitude', 'Longitude']:
    if col in df_encoded.columns:
        cols_to_drop.append(col)

X = df_encoded.drop(cols_to_drop, axis=1, errors='ignore')
y = df_encoded['Class']

# Train model
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
model = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
model.fit(X_poly, y)

# Prepare prediction input
temp_input = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)
temp_input['Payload Mass (kg)'] = payload
if f'Orbit_{orbit}' in temp_input.columns:
    temp_input[f'Orbit_{orbit}'] = 1
if f'Weather_{weather}' in temp_input.columns:
    temp_input[f'Weather_{weather}'] = 1

input_poly = poly.transform(temp_input)
prediction = model.predict(input_poly)[0]

# Output
st.metric("Predicted Launch Success Probability", f"{prediction:.2f}")
st.caption(f"Average R² Score (5-fold CV): {np.mean(scores):.2f}")
