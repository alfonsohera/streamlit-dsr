import plotly.express as px

def create_scatter_plot(filtered_df, selected_year):
    # Ensure the DataFrame is not empty before plotting
    if filtered_df.empty:
        return None

    # Create scatter plot
    fig = px.scatter(
        filtered_df,
        x="GDP per capita",
        y="Life Expectancy (IHME)",
        hover_name="country",
        size="headcount_ratio_upper_mid_income_povline",
        color="country",
        log_x=True,
        title=f"GDP per Capita vs Life Expectancy ({selected_year})",
        labels={"GDP per capita": "GDP per Capita (Log Scale)", "Life Expectancy (IHME)": "Life Expectancy"},
    )
    return fig