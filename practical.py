import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from plots import create_scatter_plot
from model import model_predict
import pickle
from sklearn.metrics import mean_squared_error

url = "https://raw.githubusercontent.com/JohannaViktor/streamlit_practical/refs/heads/main/global_development_data.csv"

df = pd.read_csv(url)

st.title("Worldwide Analysis of Quality of Life and Economic Factors")
st.write("This app enables you to explore the relationships between poverty, life expectancy, and GDP across various countries and years. Use the panels to select options and interact with the data.")

# Global selection of year with a sidebar
overall_min_year, overall_max_year = int(df["year"].min()), int(df["year"].max())
selected_year = st.sidebar.slider(
    "Select Year",
    min_value=overall_min_year,
    max_value=overall_max_year,
    value=overall_max_year
)

tab1, tab2, tab3, tab4 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer", "Predictions"])

# Global Overview Tab
with tab1:
    st.subheader("Global Overview")

    # Filter dataset based on selected year
    filtered_df = df[df["year"] == selected_year]

    # Calculate required statistics
    mean_life_exp = filtered_df["Life Expectancy (IHME)"].mean()
    median_gdp_per_capita = filtered_df["GDP per capita"].median()
    mean_upper_mid_income_povline = filtered_df["headcount_ratio_upper_mid_income_povline"].mean()
    num_countries = filtered_df["country"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Mean Life Expectancy", value=round(mean_life_exp, 2))
    with col2:
        st.metric(label="Median GDP per Capita", value=f"${round(median_gdp_per_capita, 2):,}")
    with col3:
        st.metric(label="Mean Upper-Mid Income Poverty Ratio", value=round(mean_upper_mid_income_povline, 2))
    with col4:
        st.metric(label="Number of Countries", value=num_countries)

    # Create scatter plot for Global Overview
    fig = create_scatter_plot(filtered_df, selected_year)
    st.plotly_chart(fig, use_container_width=True)

# Country Deep Dive Tab
with tab2:
    st.subheader("Country Deep Dive")

    countrieslist = df["country"].dropna().unique().tolist()
    country = st.selectbox("Select Country", countrieslist)

    # Filter dataset for the selected country
    filtered_df = df[df["country"] == country]

    # Display latest statistics for selected country
    st.subheader("Latest Statistics")
    if not filtered_df.empty:
        latest_values = filtered_df.sort_values("year").groupby("country").last().reset_index()
        for _, row in latest_values.iterrows():
            st.subheader(f"Latest Data for {row['country']}")
            st.write(f"**Life Expectancy (IHME):** {row['Healthy Life Expectancy (IHME)']:.2f} years")
            st.write(f"**GDP per Capita:** ${row['GDP per capita']:,.2f}")
            st.markdown("---")
    else:
        st.warning("No data available for the selected country.")

    # Filter dataset for the selected year
    year_filtered_df = filtered_df[filtered_df["year"] == selected_year]

    # Plot Life Expectancy and GDP per Capita over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df["year"],
        y=filtered_df["Healthy Life Expectancy (IHME)"],
        mode="lines+markers",
        name="Life Expectancy",
        line=dict(width=2, color="blue"),
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df["year"],
        y=filtered_df["GDP per capita"],
        mode="lines+markers",
        name="GDP per Capita",
        line=dict(width=2, color="red"),
        yaxis="y2"
    ))

    fig.update_layout(
        title=f"Life Expectancy & GDP per Capita in {country}",
        xaxis_title="Year",
        yaxis=dict(title="Life Expectancy (Years)", side="left"),
        yaxis2=dict(
            title="GDP per Capita",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(x=0.1, y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display statistics for selected year
    if not year_filtered_df.empty:
        row = year_filtered_df.iloc[0]
        st.subheader(f"Data for {row['country']} in {selected_year}")
        st.write(f"**Life Expectancy (IHME):** {row['Healthy Life Expectancy (IHME)']:.2f} years")
        st.write(f"**GDP per Capita:** ${row['GDP per capita']:,.2f}")
    else:
        st.warning(f"No data available for the selected year ({selected_year}).")

    # Histograms for the selected country and year
    features = ["GDP per capita", "Healthy Life Expectancy (IHME)", "headcount_ratio_upper_mid_income_povline"]
    for feature in features:
        fig = go.Figure()
        year_filtered_df_world = df[df["year"] == selected_year]

        # World distribution for the feature in the selected year
        fig.add_trace(go.Histogram(
            x=year_filtered_df_world[feature],
            nbinsx=30,  
            name=f"World Distribution of {feature}",
            marker_color='lightblue',
            opacity=0.7,
            histnorm='probability density'  # Normalized histogram to show relative frequencies
        ))

        # Add the red triangle for the selected country
        fig.add_trace(go.Scatter(
            x=[year_filtered_df[feature].iloc[0]],  # First row's feature value
            y=[0],  # Set y to 0 to show the triangle at the bottom
            mode='markers',
            name=f"{country} ({feature})",
            marker=dict(
                symbol='triangle-up',
                color='red',
                size=12
            ),
            showlegend=True
        ))
        fig.update_layout(
            title=f"{feature} Distribution in {country} ({selected_year})",
            xaxis_title=feature,
            yaxis_title="Density",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            showlegend=True,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

# Data Explorer Tab
with tab3:
    st.subheader("Data Explorer")

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
        df["country"].dropna().unique().tolist(),
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

# Predictions Tab
with tab4:
    features = ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']

    # Allow user to input feature values
    gdp_per_capita = st.number_input("Enter GDP per capita", min_value=0, step=100, format="%d")
    headcount_ratio = st.number_input("Enter Headcount Ratio (Upper-Mid Income Poverty Line)", min_value=0.0, step=0.01, format="%.2f")
    year = st.number_input("Enter Year", min_value=2000, max_value=2023, step=1)
    
    # Initialize variables
    predicted_value = None
    mse = None
    importance_df = pd.DataFrame()

    # Button to make prediction
    if st.button("Predict Life Expectancy"):
        predicted_value, mse, importance_df = model_predict(model, gdp_per_capita, headcount_ratio, year, features)

    # Display the prediction result only if prediction was done
    if predicted_value is not None:
        st.subheader("Prediction Result")
        st.write(f"Predicted Life Expectancy: {predicted_value:.2f} years")

    if mse is not None:
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")

    # Display feature importance only if prediction was done
    if not importance_df.empty:
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title='Feature Importance (Random Forest)', labels={'Importance': 'Importance', 'Feature': 'Feature'})
        st.subheader("Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
