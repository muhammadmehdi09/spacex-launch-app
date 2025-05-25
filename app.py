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

# Extract year from 'Date' column if available
date_column = next((col for col in df.columns if 'date' in col.lower()), None)
if date_column:
    df['Year'] = pd.to_datetime(df[date_column]).dt.year
else:
    df['Year'] = 0  # fallback if date column missing

# Sidebar - Filter
st.sidebar.header("🔍 Filter Launch Data")
year_filter = st.sidebar.selectbox("Select Launch Year:", options=sorted(df['Year'].unique()), index=0)
site_filter = st.sidebar.selectbox("Select Launch Site:", options=['All'] + sorted(df['Launch Site'].unique()), index=0)

# Filter data
filtered_df = df[df['Year'] == year_filter]
if site_filter != 'All':
    filtered_df = filtered_df[filtered_df['Launch Site'] == site_filter]

# Title
st.title("🚀 SpaceX Launch Analysis & Prediction Platform")
st.markdown("Explore historical launches and predict success using machine learning.")

# Show data
with st.expander("📄 View Filtered Data"):
    st.dataframe(filtered_df)

# Success by Orbit
st.subheader("📊 Success Rate by Orbit")
if not filtered_df.empty:
    orbit_success = filtered_df.groupby("Orbit")['Class'].mean().sort_values(ascending=False)
    st.bar_chart(orbit_success)
else:
    st.info("No data for selected filter.")

# Map
st.subheader("🗺️ Launch Sites Map")
if 'Latitude' in df.columns and 'Longitude' in df.columns:
    map_df = filtered_df[['Launch Site', 'Latitude', 'Longitude', 'Class']].drop_duplicates()
    launch_map = folium.Map(location=[map_df['Latitude'].mean(), map_df['Longitude'].mean()], zoom_start=3)
    for _, row in map_df.iterrows():
        color = "green" if row['Class'] == 1 else "red"
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=row['Launch Site'],
            icon=folium.Icon(color=color)
        ).add_to(launch_map)
    folium_static(launch_map)
else:
    st.warning("Map data missing Latitude/Longitude columns.")

# Machine Learning Prediction
st.subheader("🤖 Predict Launch Success")
payload = st.slider("Payload Mass (kg):", int(df['Payload Mass (kg)'].min()), int(df['Payload Mass (kg)'].max()))
orbit = st.selectbox("Orbit:", sorted(df['Orbit'].dropna().unique()))
weather = st.selectbox("Weather:", sorted(df['Weather'].dropna().unique()))

# One-hot encode input
df_encoded = pd.get_dummies(df, columns=['Orbit', 'Weather'], drop_first=True)
X = df_encoded.drop(['Class', 'Year', 'Launch Site'] + ([date_column] if date_column else []) + [col for col in ['Latitude', 'Longitude'] if col in df.columns], axis=1, errors='ignore')
y = df_encoded['Class']

# Fit polynomial model
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
model = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
model.fit(X_poly, y)

# Prepare input
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
