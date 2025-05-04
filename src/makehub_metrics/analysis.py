#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import os
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# Set plot style
plt.style.use("ggplot")
sns.set(font_scale=1.2)

# Constants
DEFAULT_INPUT_CSV = "makehub_raw_data.csv"
DEFAULT_OUTPUT_DIR = "makehub_analysis"


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load and clean the raw MakeHub data

    Args:
        csv_path: Path to the CSV file

    Returns:
        Cleaned DataFrame
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Convert timestamp strings to datetime objects
    timestamp_columns = ["timestamp", "request_start_time", "request_end_time"]
    for col in timestamp_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Convert string boolean values to actual booleans
    if "success" in df.columns:
        df["success"] = df["success"].map({"TRUE": True, "FALSE": False})

    # Calculate request duration from timestamps as a verification
    if "request_start_time" in df.columns and "request_end_time" in df.columns:
        df["calculated_duration"] = (
            df["request_end_time"] - df["request_start_time"]
        ).dt.total_seconds()

        # Check for discrepancies between measured and calculated latency
        df["latency_discrepancy"] = abs(
            df["actual_latency_sec"] - df["calculated_duration"]
        )
        discrepancy_threshold = 0.1  # 100ms
        suspicious_rows = df[df["latency_discrepancy"] > discrepancy_threshold]
        if not suspicious_rows.empty:
            print(
                f"Warning: Found {len(suspicious_rows)} rows with significant latency discrepancies."
            )

    # Convert MakeHub latency from ms to sec for easier comparison
    if "makehub_avg_latency_ms" in df.columns:
        df["makehub_avg_latency_sec"] = df["makehub_avg_latency_ms"] / 1000.0

    # Calculate the difference between MakeHub and actual latency
    if "makehub_avg_latency_sec" in df.columns and "actual_latency_sec" in df.columns:
        df["latency_diff_sec"] = (
            df["actual_latency_sec"] - df["makehub_avg_latency_sec"]
        )
        df["latency_diff_percent"] = (
            df["latency_diff_sec"] / df["makehub_avg_latency_sec"]
        ) * 100

    # Drop rows with missing essential data
    essential_columns = ["region_id", "makehub_window_minutes", "actual_latency_sec"]
    df = df.dropna(subset=essential_columns)

    print(f"Data loaded successfully. {len(df)} valid rows.")
    return df


def analyze_regional_performance(df: pd.DataFrame, output_dir: str):
    """Analyze performance by region

    Args:
        df: The DataFrame with MakeHub data
        output_dir: Directory to save output files
    """
    print("Analyzing regional performance...")

    # Group by region and calculate metrics
    region_metrics = df.groupby("region_id").agg(
        {
            "actual_latency_sec": ["mean", "std", "min", "max", "count"],
            "time_to_first_token_sec": ["mean", "std"],
            "makehub_avg_latency_sec": ["mean", "std"],
            "latency_diff_sec": ["mean", "std"],
            "latency_diff_percent": ["mean", "std"],
            "success": ["mean", "count"],
        }
    )

    # Flatten the hierarchical index
    region_metrics.columns = [
        "_".join(col).strip() for col in region_metrics.columns.values
    ]
    region_metrics.reset_index(inplace=True)

    # Rename success_mean to success_rate for clarity
    region_metrics = region_metrics.rename(columns={"success_mean": "success_rate"})

    # Sort by mean actual latency
    region_sorted = region_metrics.sort_values("actual_latency_sec_mean")

    # Create a bar plot comparing actual vs MakeHub latency by region
    plt.figure(figsize=(14, 8))

    # Set up the bar positions
    regions = region_sorted["region_id"].tolist()
    x = np.arange(len(regions))
    width = 0.35

    # Create the bars
    plt.bar(
        x - width / 2,
        region_sorted["actual_latency_sec_mean"],
        width,
        label="Actual Latency",
        alpha=0.7,
        color="blue",
    )
    plt.bar(
        x + width / 2,
        region_sorted["makehub_avg_latency_sec_mean"],
        width,
        label="MakeHub Latency",
        alpha=0.7,
        color="green",
    )

    # Add error bars
    plt.errorbar(
        x - width / 2,
        region_sorted["actual_latency_sec_mean"],
        yerr=region_sorted["actual_latency_sec_std"],
        fmt="none",
        color="black",
        alpha=0.5,
    )
    plt.errorbar(
        x + width / 2,
        region_sorted["makehub_avg_latency_sec_mean"],
        yerr=region_sorted["makehub_avg_latency_sec_std"],
        fmt="none",
        color="black",
        alpha=0.5,
    )

    # Customize the plot
    plt.xlabel("Azure Region")
    plt.ylabel("Latency (seconds)")
    plt.title("Actual vs MakeHub Latency by Region")
    plt.xticks(x, regions, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_dir, "regional_latency_comparison.png"))
    plt.close()

    # Create a table with the region metrics
    region_metrics.to_csv(
        os.path.join(output_dir, "regional_performance_metrics.csv"), index=False
    )

    # Create a correlation heatmap for regional metrics
    metrics_cols = [
        col
        for col in region_metrics.columns
        if col not in ["region_id", "success_count", "actual_latency_sec_count"]
    ]
    corr_matrix = region_metrics[metrics_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, fmt=".2f"
    )
    plt.title("Correlation between Regional Performance Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regional_metrics_correlation.png"))
    plt.close()

    print(f"Regional analysis complete. Results saved to {output_dir}")

    return region_metrics


def analyze_window_impact(df: pd.DataFrame, output_dir: str):
    """Analyze impact of different MakeHub window sizes

    Args:
        df: The DataFrame with MakeHub data
        output_dir: Directory to save output files
    """
    print("Analyzing MakeHub window size impact...")

    # Check if we have multiple window sizes
    window_sizes = df["makehub_window_minutes"].unique()
    if len(window_sizes) <= 1:
        print("  Warning: Only one window size found. Skipping window impact analysis.")
        return None

    # Group by window size and calculate metrics
    window_metrics = df.groupby("makehub_window_minutes").agg(
        {
            "actual_latency_sec": ["mean", "std"],
            "makehub_avg_latency_sec": ["mean", "std"],
            "latency_diff_sec": ["mean", "std"],
            "latency_diff_percent": ["mean", "std"],
        }
    )

    # Also calculate absolute metrics separately
    abs_metrics = df.groupby("makehub_window_minutes").agg(
        {
            "latency_diff_sec": lambda x: np.mean(np.abs(x)),
            "latency_diff_percent": lambda x: np.mean(np.abs(x)),
        }
    )

    abs_metrics.columns = [f"{col}_abs" for col in abs_metrics.columns]

    # Flatten the hierarchical index
    window_metrics.columns = [
        "_".join(col).strip() for col in window_metrics.columns.values
    ]
    window_metrics.reset_index(inplace=True)

    # Merge with absolute metrics
    abs_metrics.reset_index(inplace=True)
    window_metrics = pd.merge(window_metrics, abs_metrics, on="makehub_window_minutes")

    # Calculate root mean squared error and mean absolute error for each window
    window_error_metrics = []

    for window in window_sizes:
        window_df = df[df["makehub_window_minutes"] == window]

        # Calculate error metrics
        mse = mean_squared_error(
            window_df["actual_latency_sec"], window_df["makehub_avg_latency_sec"]
        )
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(
            window_df["actual_latency_sec"], window_df["makehub_avg_latency_sec"]
        )

        # Calculate correlation
        corr, p_value = pearsonr(
            window_df["actual_latency_sec"], window_df["makehub_avg_latency_sec"]
        )

        window_error_metrics.append(
            {
                "makehub_window_minutes": window,
                "rmse": rmse,
                "mae": mae,
                "correlation": corr,
                "correlation_p_value": p_value,
                "sample_count": len(window_df),
            }
        )

    window_error_df = pd.DataFrame(window_error_metrics)

    # Create a line plot showing accuracy metrics by window size
    plt.figure(figsize=(12, 8))

    # Plot RMSE
    plt.plot(
        window_error_df["makehub_window_minutes"],
        window_error_df["rmse"],
        marker="o",
        linestyle="-",
        color="blue",
        label="RMSE",
    )

    # Plot MAE
    plt.plot(
        window_error_df["makehub_window_minutes"],
        window_error_df["mae"],
        marker="s",
        linestyle="-",
        color="green",
        label="MAE",
    )

    # Add correlation on secondary axis
    ax2 = plt.twinx()
    ax2.plot(
        window_error_df["makehub_window_minutes"],
        window_error_df["correlation"],
        marker="^",
        linestyle="-",
        color="red",
        label="Correlation",
    )
    ax2.set_ylabel("Correlation", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 1)

    # Customize the plot
    plt.xlabel("MakeHub Window Size (minutes)")
    plt.ylabel("Error (seconds)")
    plt.title("Accuracy of MakeHub Latency by Window Size")
    plt.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "window_size_impact.png"))
    plt.close()

    # Create a bar chart comparing actual vs. MakeHub latency by window size
    plt.figure(figsize=(10, 6))

    # Set up the bar positions
    windows = window_metrics["makehub_window_minutes"].tolist()
    x = np.arange(len(windows))
    width = 0.35

    # Create the bars
    plt.bar(
        x - width / 2,
        window_metrics["actual_latency_sec_mean"],
        width,
        label="Actual Latency",
        alpha=0.7,
        color="blue",
    )
    plt.bar(
        x + width / 2,
        window_metrics["makehub_avg_latency_sec_mean"],
        width,
        label="MakeHub Latency",
        alpha=0.7,
        color="green",
    )

    # Add error bars
    plt.errorbar(
        x - width / 2,
        window_metrics["actual_latency_sec_mean"],
        yerr=window_metrics["actual_latency_sec_std"],
        fmt="none",
        color="black",
        alpha=0.5,
    )
    plt.errorbar(
        x + width / 2,
        window_metrics["makehub_avg_latency_sec_mean"],
        yerr=window_metrics["makehub_avg_latency_sec_std"],
        fmt="none",
        color="black",
        alpha=0.5,
    )

    # Customize the plot
    plt.xlabel("MakeHub Window Size (minutes)")
    plt.ylabel("Latency (seconds)")
    plt.title("Actual vs MakeHub Latency by Window Size")
    plt.xticks(x, windows)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "window_size_latency_comparison.png"))
    plt.close()

    # Save the metrics to CSV
    combined_metrics = pd.merge(
        window_metrics, window_error_df, on="makehub_window_minutes"
    )
    combined_metrics.to_csv(
        os.path.join(output_dir, "window_size_metrics.csv"), index=False
    )

    print(f"Window size analysis complete. Results saved to {output_dir}")

    return combined_metrics


def analyze_time_series(df: pd.DataFrame, output_dir: str):
    """Analyze time series patterns in the data

    Args:
        df: The DataFrame with MakeHub data
        output_dir: Directory to save output files
    """
    print("Analyzing time series patterns...")

    # Check if we have timestamp data
    if "request_start_time" not in df.columns:
        print("  Warning: No timestamp data found. Skipping time series analysis.")
        return

    # Sort by timestamp
    df_sorted = df.sort_values("request_start_time")

    # Create a plot showing latency over time for different regions
    plt.figure(figsize=(15, 10))

    regions = df["region_id"].unique()
    for region in regions:
        region_df = df_sorted[df_sorted["region_id"] == region]
        plt.plot(
            region_df["request_start_time"],
            region_df["actual_latency_sec"],
            marker="o",
            linestyle="-",
            alpha=0.7,
            label=region,
        )

    plt.xlabel("Timestamp")
    plt.ylabel("Actual Latency (seconds)")
    plt.title("Latency Over Time by Region")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Format the x-axis to show reasonable time intervals
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_time_series.png"))
    plt.close()

    # Create a plot comparing actual vs MakeHub latency over time (average across regions)
    time_metrics = (
        df_sorted.groupby(pd.Grouper(key="request_start_time", freq="1min"))
        .agg(
            {
                "actual_latency_sec": "mean",
                "makehub_avg_latency_sec": "mean",
                "latency_diff_sec": "mean",
            }
        )
        .reset_index()
    )

    plt.figure(figsize=(15, 10))

    plt.plot(
        time_metrics["request_start_time"],
        time_metrics["actual_latency_sec"],
        marker="o",
        linestyle="-",
        color="blue",
        label="Actual Latency",
    )
    plt.plot(
        time_metrics["request_start_time"],
        time_metrics["makehub_avg_latency_sec"],
        marker="s",
        linestyle="-",
        color="green",
        label="MakeHub Latency",
    )

    plt.xlabel("Timestamp")
    plt.ylabel("Latency (seconds)")
    plt.title("Actual vs MakeHub Latency Over Time (Average Across Regions)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Format the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_comparison_time_series.png"))
    plt.close()

    print(f"Time series analysis complete. Results saved to {output_dir}")


def generate_optimization_recommendations(
    df: pd.DataFrame,
    window_metrics: pd.DataFrame,
    region_metrics: pd.DataFrame,
    output_dir: str,
):
    """Generate recommendations for optimizing MakeHub scoring

    Args:
        df: The DataFrame with MakeHub data
        window_metrics: DataFrame with window size metrics
        region_metrics: DataFrame with regional metrics
        output_dir: Directory to save output files
    """
    print("Generating optimization recommendations...")

    recommendations = []

    # 1. Analyze optimal window size based on accuracy metrics
    if window_metrics is not None and not window_metrics.empty:
        # Find window with lowest RMSE
        best_rmse_window = window_metrics.loc[window_metrics["rmse"].idxmin()]

        # Find window with highest correlation
        best_corr_window = window_metrics.loc[window_metrics["correlation"].idxmax()]

        recommendations.append(
            {
                "category": "Window Size",
                "recommendation": f"Use {int(best_rmse_window['makehub_window_minutes'])} minute window for minimum error (RMSE: {best_rmse_window['rmse']:.4f})",
                "supporting_data": f"RMSE values by window: {window_metrics[['makehub_window_minutes', 'rmse']].to_dict('records')}",
            }
        )

        if int(best_rmse_window["makehub_window_minutes"]) != int(
            best_corr_window["makehub_window_minutes"]
        ):
            recommendations.append(
                {
                    "category": "Window Size",
                    "recommendation": f"Alternatively, use {int(best_corr_window['makehub_window_minutes'])} minute window for best correlation with actual latency (Correlation: {best_corr_window['correlation']:.4f})",
                    "supporting_data": f"Correlation values by window: {window_metrics[['makehub_window_minutes', 'correlation']].to_dict('records')}",
                }
            )

    # 2. Analyze regional bias in MakeHub measurements
    region_bias = region_metrics.copy()
    region_bias["abs_latency_diff_percent_mean"] = abs(
        region_bias["latency_diff_percent_mean"]
    )
    region_bias = region_bias.sort_values(
        "abs_latency_diff_percent_mean", ascending=False
    )

    biased_regions = region_bias[
        region_bias["abs_latency_diff_percent_mean"] > 20
    ].copy()  # Regions with >20% difference

    if not biased_regions.empty:
        bias_text = ""
        for _, row in biased_regions.iterrows():
            direction = (
                "underestimated"
                if row["latency_diff_percent_mean"] > 0
                else "overestimated"
            )
            bias_text += f"\n- {row['region_id']}: MakeHub {direction} by {abs(row['latency_diff_percent_mean']):.1f}%"

        recommendations.append(
            {
                "category": "Regional Bias",
                "recommendation": f"Apply correction factors to these regions with significant bias: {bias_text}",
                "supporting_data": "See regional_performance_metrics.csv for full details",
            }
        )

    # 3. Throughput vs Latency analysis
    if "makehub_avg_throughput_tokens_per_sec" in df.columns:
        corr_throughput_actual = (
            df.groupby("region_id")
            .apply(
                lambda x: (
                    pearsonr(
                        x["makehub_avg_throughput_tokens_per_sec"],
                        x["actual_latency_sec"],
                    )[0]
                    if len(x) > 2
                    else np.nan
                )
            )
            .dropna()
        )

        if not corr_throughput_actual.empty:
            avg_correlation = np.mean(corr_throughput_actual)
            if abs(avg_correlation) > 0.5:
                relation = (
                    "inversely related" if avg_correlation < 0 else "directly related"
                )
                recommendations.append(
                    {
                        "category": "Throughput-Latency",
                        "recommendation": f"Incorporate throughput metrics into latency prediction models (correlation: {avg_correlation:.2f})",
                        "supporting_data": f"Throughput and latency are {relation} with correlation of {avg_correlation:.2f}",
                    }
                )

    # 4. TTFT vs Total Latency analysis
    ttft_correlation = (
        df.groupby("region_id")
        .apply(
            lambda x: (
                pearsonr(x["time_to_first_token_sec"], x["actual_latency_sec"])[0]
                if len(x) > 2
                else np.nan
            )
        )
        .dropna()
    )

    if not ttft_correlation.empty:
        avg_ttft_corr = np.mean(ttft_correlation)
        if avg_ttft_corr > 0.7:
            ttft_ratio = (
                df["time_to_first_token_sec"].mean() / df["actual_latency_sec"].mean()
            )
            recommendations.append(
                {
                    "category": "TTFT Optimization",
                    "recommendation": f"Focus on Time-To-First-Token as a key metric (avg {ttft_ratio:.1%} of total latency)",
                    "supporting_data": f"TTFT strongly correlates with total latency (r={avg_ttft_corr:.2f})",
                }
            )

    # 5. Temporal patterns
    time_variance = (
        df.groupby([pd.Grouper(key="request_start_time", freq="5min"), "region_id"])[
            "actual_latency_sec"
        ]
        .std()
        .reset_index()
    )
    high_variance_periods = time_variance[
        time_variance["actual_latency_sec"]
        > time_variance["actual_latency_sec"].mean()
        + time_variance["actual_latency_sec"].std()
    ]

    if not high_variance_periods.empty:
        recommendations.append(
            {
                "category": "Temporal Patterns",
                "recommendation": "Use adaptive window sizes during periods of high latency variance",
                "supporting_data": f"Identified {len(high_variance_periods)} time periods with high latency variance",
            }
        )

    # 6. Generate weighted scoring formula
    if window_metrics is not None and not window_metrics.empty:
        # Create weights inversely proportional to RMSE
        inverse_rmse = 1 / window_metrics["rmse"]
        weights = inverse_rmse / inverse_rmse.sum()

        formula = "Optimized Score = "
        for i, row in window_metrics.iterrows():
            window = int(row["makehub_window_minutes"])
            weight = weights.iloc[i]
            formula += f"{weight:.2f} × (Window{window}min) + "

        formula = formula[:-3]  # Remove the last " + "

        recommendations.append(
            {
                "category": "Scoring Formula",
                "recommendation": "Use a weighted combination of multiple window sizes",
                "supporting_data": formula,
            }
        )

    # 7. Global recommendations
    overall_rmse = np.sqrt(
        mean_squared_error(df["actual_latency_sec"], df["makehub_avg_latency_sec"])
    )
    overall_bias = (
        (df["actual_latency_sec"].mean() - df["makehub_avg_latency_sec"].mean())
        / df["makehub_avg_latency_sec"].mean()
        * 100
    )

    bias_direction = "underestimates" if overall_bias > 0 else "overestimates"
    recommendations.append(
        {
            "category": "Global Adjustment",
            "recommendation": f"Apply a global correction factor of {1 + overall_bias/100:.2f}× to MakeHub latency values",
            "supporting_data": f"MakeHub typically {bias_direction} latency by {abs(overall_bias):.1f}% (RMSE: {overall_rmse:.4f}s)",
        }
    )

    # Save recommendations to file
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv(
        os.path.join(output_dir, "optimization_recommendations.csv"), index=False
    )

    # Generate a formatted text report
    with open(os.path.join(output_dir, "recommendations_report.txt"), "w") as f:
        f.write("# MakeHub Optimization Recommendations\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: {args.input}\n")
        f.write(f"Records Analyzed: {len(df)}\n\n")

        current_category = None
        for _, row in recommendations_df.iterrows():
            if row["category"] != current_category:
                current_category = row["category"]
                f.write(f"\n## {current_category}\n\n")

            f.write(f"* **Recommendation**: {row['recommendation']}\n")
            f.write(f"  * *Supporting Data*: {row['supporting_data']}\n\n")

        f.write("\n## Summary\n\n")
        f.write(
            f"Overall, MakeHub latency measurements have an RMSE of {overall_rmse:.4f} seconds compared to actual measurements.\n"
        )
        f.write(
            f"MakeHub {bias_direction} actual latency by approximately {abs(overall_bias):.1f}% on average.\n"
        )

        if window_metrics is not None and not window_metrics.empty:
            best_window = int(
                window_metrics.loc[window_metrics["rmse"].idxmin()][
                    "makehub_window_minutes"
                ]
            )
            f.write(
                f"The optimal window size for MakeHub metrics appears to be {best_window} minutes based on this analysis.\n"
            )

    print(f"Optimization recommendations generated. Results saved to {output_dir}")


def main(args):
    """Main function for MakeHub data analysis

    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load and clean the data
    df = load_and_clean_data(args.input)

    # Analyze regional performance
    region_metrics = analyze_regional_performance(df, args.output)

    # Analyze window impact
    window_metrics = analyze_window_impact(df, args.output)

    # Analyze time series patterns
    analyze_time_series(df, args.output)

    # Generate optimization recommendations
    generate_optimization_recommendations(
        df, window_metrics, region_metrics, args.output
    )

    print("\nAnalysis complete! All results saved to:", args.output)
    print("Key findings and recommendations can be found in:")
    print(f"  - {os.path.join(args.output, 'recommendations_report.txt')}")
    print(f"  - {os.path.join(args.output, 'regional_performance_metrics.csv')}")
    if window_metrics is not None:
        print(f"  - {os.path.join(args.output, 'window_size_metrics.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze MakeHub raw data and generate optimization recommendations"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_CSV,
        help=f"Input CSV file path (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for analysis results (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    main(args)
