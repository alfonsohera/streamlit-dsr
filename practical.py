import streamlit as st
import pandas as pd
import requests

url = "https://raw.githubusercontent.com/JohannaViktor/streamlit_practical/refs/heads/main/global_development_data.csv"
df = pd.read_csv(url)

st.title("Worldwide Analysis of Quality of Life and Economic Factors")
st.write("This app enables you to explore the relationships between poverty, \
            life expectancy, and GDP across various countries and years. \
            Use the panels to select options and interact with the data.")
tab1,tab2,tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])

with tab1:
    pass
with tab2:
    pass
with tab3:
    unique_countries = df["country"].dropna().unique().tolist()

    # Get overall min and max years
    overall_min_year, overall_max_year = int(df["year"].min()), int(df["year"].max())

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

    