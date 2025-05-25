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

# Sidebar - Filter
st.sidebar.header("🔍 Filter Launch Data")
year_filter = st.sidebar.selectbox("Select Launch Year:", options=sorted(df['Year'].unique()), index=0)
site_filter = st.sidebar.selectbox("Select Launch Site:", options=['All'] + sorted(df['Launch Site'].unique()), index=0)

# Filter data based on selection
filtered_df = df[df['Year'] == year_filter]
if site_filter != 'All':
    filtered_df = filtered_df[filtered_df['Launch Site'] == site_filter]

# Title
st.title("🚀 SpaceX Launch Analysis & Prediction Platform")
st.markdown("Analyze historical launches and predict success with machine learning.")

# Show data preview
with st.expander("📄 View Filtered Data"):
    st.dataframe(filtered_df)

# Visualization - Success Rate by Orbit
st.subheader("📊 Success Rate by Orbit")
orbit_success = filtered_df.groupby("Orbit")['Class'].mean().sort_values(ascending=False)
st.bar_chart(orbit_success)

# Map - Launch Locations
st.subheader("🗺️ Launch Sites Map")
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

# ML Section
st.subheader("🤖 Predict Launch Success")

# Input features
payload = st.slider("Payload Mass (kg):", int(df['Payload Mass (kg)'].min()), int(df['Payload Mass (kg)'].max()))
orbit = st.selectbox("Orbit:", sorted(df['Orbit'].unique()))
weather = st.selectbox("Weather:", sorted(df['Weather'].unique()))

# Prepare input data
input_df = pd.DataFrame({
    'Payload Mass (kg)': [payload],
    'Orbit_' + orbit: [1],
    'Weather_' + weather: [1]
})

# Match training data structure
df_encoded = pd.get_dummies(df, columns=['Orbit', 'Weather'], drop_first=True)
X = df_encoded.drop(['Class', 'Year', 'Launch Site', 'Latitude', 'Longitude'], axis=1)
y = df_encoded['Class']

# Polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')

# Fit and predict
model.fit(X_poly, y)

# Prepare input with same structure
input_full = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)
for col in input_df.columns:
    if col in input_full.columns:
        input_full[col] = input_df[col].values
input_poly = poly.transform(input_full)

prediction = model.predict(input_poly)[0]

# Show prediction
st.metric(label="Predicted Launch Success Probability", value=f"{prediction:.2f}")
st.caption(f"Average R² Score from CV: {np.mean(scores):.2f}")
