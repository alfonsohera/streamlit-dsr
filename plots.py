import plotly.express as px

def create_scatter_plot(filtered_df, selected_year):
    # Ensure the DataFrame is not empty before plotting
    if filtered_df.empty:
        return None

    # Normalize bubble size manually
    min_bubble_size = 5
    max_bubble_size = 50

    # Avoid division by zero if population values are constant
    if filtered_df["Population"].max() != filtered_df["Population"].min():
        filtered_df["bubble_size"] = (
            (filtered_df["Population"] - filtered_df["Population"].min()) /
            (filtered_df["Population"].max() - filtered_df["Population"].min())
        ) * (max_bubble_size - min_bubble_size) + min_bubble_size
    else:
        filtered_df["bubble_size"] = min_bubble_size  # Assign minimum size if all values are the same

    # Create scatter plot
    fig = px.scatter(
        filtered_df,
        x="GDP per capita",
        y="Life Expectancy (IHME)",
        hover_name="country",
        size="bubble_size",
        color="country",
        log_x=True,
        title=f"GDP per Capita vs Life Expectancy ({selected_year})",
        labels={"GDP per capita": "GDP per Capita (Log Scale)", "Life Expectancy (IHME)": "Life Expectancy"},
        size_max=max_bubble_size
    )
    return fig