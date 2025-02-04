import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from plots import create_scatter_plot

url = "https://raw.githubusercontent.com/JohannaViktor/streamlit_practical/refs/heads/main/global_development_data.csv"

df = pd.read_csv(url)

st.title("Worldwide Analysis of Quality of Life and Economic Factors")
st.write("This app enables you to explore the relationships between poverty, \
            life expectancy, and GDP across various countries and years. \
            Use the panels to select options and interact with the data.")
tab1,tab2,tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])
overall_min_year, overall_max_year = int(df["year"].min()), int(df["year"].max())
with tab1:
    
    #Slider to select year
    st.subheader("Global overview")
    selected_year = st.slider(
        "Select year range for visualization:",
        min_value=overall_min_year,
        max_value=overall_max_year,
        value=(overall_min_year))
    # Filter dataset based on selected year range and countries
    filtered_df = df[(df["year"] == selected_year)]
    col1,col2,col3,col4  = st.columns(4)

    # Calculate required statistics
    mean_life_exp = filtered_df["Life Expectancy (IHME)"].mean()
    median_gdp_per_capita = filtered_df["GDP per capita"].median()
    mean_upper_mid_income_povline = filtered_df["headcount_ratio_upper_mid_income_povline"].mean()
    num_countries = filtered_df["country"].nunique()

    #Metric tabs
    with col1:
         st.metric(label="Mean Life Expectancy", value=round(mean_life_exp, 2))
    with col2:
        st.metric(label="Median GDP per Capita", value=f"${round(median_gdp_per_capita, 2):,}")
    with col3:
        st.metric(label="Mean Upper-Mid Income Poverty Ratio", value=round(mean_upper_mid_income_povline, 2))
    with col4: 
        st.metric(label="Number of Countries", value=num_countries)
    pass
    
    # Create scatter plot
    fig = create_scatter_plot(filtered_df, selected_year)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    pass
with tab3:
    unique_countries = df["country"].dropna().unique().tolist()

    # Get overall min and max years
    st.subheader("Data Explorer")
    st.write("This is the complete dataset:")

    # Allow user to define min and max year range
    selected_year_range = st.slider(
        "Select year range",
        min_value=overall_min_year,
        max_value=overall_max_year,
        value=(overall_min_year, overall_max_year),  # Default to full range
    )

    # Country multiselect
    selected_countries = st.multiselect(
        "Select countries",
        unique_countries,
    )

    # Filter dataset based on selected year range and countries
    filtered_df = df[
        (df["year"].between(*selected_year_range)) & (df["country"].isin(selected_countries))
    ]

    st.dataframe(filtered_df)

    # Convert filtered DataFrame to CSV
    csv = filtered_df.to_csv(index=False).encode("utf-8")

    # Download button
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"filtered_data_{selected_year_range[0]}_{selected_year_range[1]}.csv",
        mime="text/csv"
    )