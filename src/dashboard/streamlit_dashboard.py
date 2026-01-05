import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import time
import json
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Intrusion Detection System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .alert-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class Dashboard:
    """Real-time intrusion detection dashboard."""

    def __init__(self, logs_dir: str = "logs"):
        """Initialize dashboard with logs directory.
        Args: logs_dir (str) - Directory containing log files.
        """
        script_dir = Path(__file__).parent
        self.logs_dir = script_dir.parent / logs_dir
        self.detections_file = self.logs_dir / "detections.csv"
        self.performance_file = self.logs_dir / "performance.csv"
        self.alerts_file = self.logs_dir / "alerts.csv"
        self.retraining_file = self.logs_dir / "retraining.csv"
        self.summary_file = self.logs_dir / "summary.json"

    def load_summary(self) -> dict:
        """Load summary statistics from JSON. Returns: dict."""
        try:
            with open(self.summary_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def load_detections(self, limit: int = None) -> pd.DataFrame:
        """Load detection logs from CSV. Args: limit (int, optional). Returns: DataFrame."""
        try:
            df = pd.read_csv(self.detections_file)
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            if limit and not df.empty:
                df = df.tail(limit)
            return df
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()

    def load_performance(self) -> pd.DataFrame:
        """Load performance metrics from CSV. Returns: DataFrame."""
        try:
            df = pd.read_csv(self.performance_file)
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()

    def load_alerts(self, limit: int = None) -> pd.DataFrame:
        """Load alerts from CSV. Args: limit (int, optional). Returns: DataFrame."""
        try:
            df = pd.read_csv(self.alerts_file)
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            if limit and not df.empty:
                df = df.tail(limit)
            return df
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()

    def load_retraining(self) -> pd.DataFrame:
        """Load retraining logs from CSV. Returns: DataFrame."""
        try:
            df = pd.read_csv(self.retraining_file)
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()

    def render_header(self):
        """Render dashboard header with title and session info."""
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.title(" Real-Time Intrusion Detection System")

        with col2:
            summary = self.load_summary()
            if summary:
                start_time = datetime.fromisoformat(
                    summary.get("session_start", datetime.now().isoformat())
                )
                elapsed = datetime.now() - start_time
                st.metric(
                    "Session Duration",
                    f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s",
                )

        with col3:
            if st.button(" Refresh", width="stretch"):
                st.rerun()

        st.markdown("---")

    def render_metrics_overview(self):
        """Render key metrics overview (packets, attacks, alerts, retrains)."""
        summary = self.load_summary()

        if not summary:
            st.warning(
                "No data available yet. Start the detection system to see metrics."
            )
            return

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Total Packets", f"{summary.get('total_packets', 0):,}", delta=None
            )

        with col2:
            attacks = summary.get("total_attacks_detected", 0)
            attack_rate = summary.get("attack_rate_percent", 0)
            st.metric("Attacks Detected", f"{attacks:,}", delta=f"{attack_rate:.1f}%")

        with col3:
            benign = summary.get("total_benign_detected", 0)
            st.metric("Benign Traffic", f"{benign:,}", delta=None)

        with col4:
            alerts = summary.get("total_alerts", 0)
            st.metric(
                "Critical Alerts", f"{alerts:,}", delta=None, delta_color="inverse"
            )

        with col5:
            retrains = summary.get("total_retrains", 0)
            st.metric("Model Retrains", f"{retrains}", delta=None)

        st.markdown("##### Model Evolution Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            promotion_count = summary.get("promotion_count", 0)
            st.metric(
                "Promotions",
                f"{promotion_count}",
                help="Number of times candidate model was promoted to production",
            )

        with col2:
            rejected_count = summary.get("rejected_count", 0)
            st.metric(
                "Rejections",
                f"{rejected_count}",
                help="Number of times candidate model was rejected",
            )

        with col3:
            reset_count = summary.get("reset_count", 0)
            st.metric(
                "Resets",
                f"{reset_count}",
                help="Number of times candidate model was reset",
            )

        with col4:
            promotion_rate = summary.get("promotion_rate", 0)
            st.metric(
                "Promotion Rate",
                f"{promotion_rate:.1f}%",
                help="Percentage of successful promotions",
            )

    def render_performance_charts(self):
        """Render MCC and classification metrics charts with evolution."""
        st.subheader(" Model Performance Over Time")

        perf_df = self.load_performance()

        if perf_df.empty:
            st.info("No performance data available yet.")
            return

        required_cols = [
            "packets_processed",
            "accuracy",
            "mcc",
            "precision",
            "recall",
            "f1_score",
        ]
        missing_cols = [col for col in required_cols if col not in perf_df.columns]

        if missing_cols:
            st.warning(
                f"Performance data incomplete. Missing columns: {', '.join(missing_cols)}"
            )
            return

        st.markdown("#### MCC Evolution")
        mcc_initial = perf_df["mcc"].iloc[0]
        mcc_final = perf_df["mcc"].iloc[-1]
        mcc_delta = mcc_final - mcc_initial
        st.metric(
            "MCC",
            f"{mcc_final:.4f}",
            delta=f"{mcc_delta:+.4f}",
            help=f"Initial: {mcc_initial:.4f} → Final: {mcc_final:.4f}",
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=perf_df["packets_processed"],
                y=perf_df["mcc"],
                mode="lines+markers",
                name="MCC",
                line=dict(color="#3498db", width=2),
            )
        )

        fig.update_layout(
            xaxis_title="Packets Processed",
            yaxis_title="MCC",
            yaxis_range=[0, 1],
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig, config={'displayModeBar': False})

        st.markdown("#### Classification Metrics Evolution")
        col1, col2, col3, col4 = st.columns(4)

        metrics_info = [
            ("accuracy", "Accuracy", col1),
            ("precision", "Precision", col2),
            ("recall", "Recall", col3),
            ("f1_score", "F1-Score", col4),
        ]

        for metric_key, metric_name, col in metrics_info:
            with col:
                initial_val = perf_df[metric_key].iloc[0]
                final_val = perf_df[metric_key].iloc[-1]
                delta = final_val - initial_val
                st.metric(
                    metric_name,
                    f"{final_val:.4f}",
                    delta=f"{delta:+.4f}",
                    help=f"Initial: {initial_val:.4f} → Final: {final_val:.4f}",
                )

        fig = make_subplots(specs=[[{"secondary_y": False}]])

        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
        colors = ["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

        for metric, color in zip(metrics_to_plot, colors):
            fig.add_trace(
                go.Scatter(
                    x=perf_df["packets_processed"],
                    y=perf_df[metric],
                    mode="lines",
                    name=metric.replace("_", " ").title(),
                    line=dict(color=color, width=2),
                )
            )

        fig.update_layout(
            xaxis_title="Packets Processed",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=300,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, config={'displayModeBar': False})

    def render_detection_distribution(self):
        """Render prediction distribution and accuracy charts."""
        st.subheader(" Detection Distribution")

        detections = self.load_detections(limit=5000)

        if detections.empty:
            st.info("No detection data available yet.")
            return

        required_cols = ["prediction_label", "correct", "prediction"]
        missing_cols = [col for col in required_cols if col not in detections.columns]

        if missing_cols:
            st.warning(
                f"Detection data incomplete. Missing columns: {', '.join(missing_cols)}"
            )
            return

        col1, col2 = st.columns(2)

        with col1:
            pred_counts = detections["prediction_label"].value_counts()
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=pred_counts.index,
                        values=pred_counts.values,
                        hole=0.4,
                        marker=dict(colors=["#2ecc71", "#e74c3c"]),
                    )
                ]
            )
            fig.update_layout(
                title="Predictions Distribution",
                height=350,
                annotations=[
                    dict(text="Total", x=0.5, y=0.5, font_size=20, showarrow=False)
                ],
            )
            st.plotly_chart(fig, config={'displayModeBar': False})

        with col2:
            correct_counts = detections["correct"].value_counts()
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["Correct", "Incorrect"],
                        values=[
                            correct_counts.get(True, 0),
                            correct_counts.get(False, 0),
                        ],
                        hole=0.4,
                        marker=dict(colors=["#2ecc71", "#e74c3c"]),
                    )
                ]
            )
            fig.update_layout(
                title="Prediction Accuracy",
                height=350,
                annotations=[
                    dict(text="Accuracy", x=0.5, y=0.5, font_size=20, showarrow=False)
                ],
            )
            st.plotly_chart(fig, config={'displayModeBar': False})

        if "timestamp" in detections.columns and not detections.empty:
            detections_resampled = (
                detections.set_index("timestamp")
                .resample("1s")["prediction"]
                .value_counts()
                .unstack(fill_value=0)
            )
        else:
            detections_resampled = pd.DataFrame()

        if not detections_resampled.empty:
            fig = go.Figure()
            if 0 in detections_resampled.columns:
                fig.add_trace(
                    go.Scatter(
                        x=detections_resampled.index,
                        y=detections_resampled[0],
                        mode="lines",
                        name="Benign",
                        fill="tozeroy",
                        line=dict(color="#2ecc71"),
                    )
                )
            if 1 in detections_resampled.columns:
                fig.add_trace(
                    go.Scatter(
                        x=detections_resampled.index,
                        y=detections_resampled[1],
                        mode="lines",
                        name="Attack",
                        fill="tozeroy",
                        line=dict(color="#e74c3c"),
                    )
                )

            fig.update_layout(
                title="Detection Rate Over Time (packets/second)",
                xaxis_title="Time",
                yaxis_title="Packets",
                height=300,
                hovermode="x unified",
            )
            st.plotly_chart(fig, config={'displayModeBar': False})

    def render_attack_type_distribution(self):
        """Render attack type distribution charts and statistics."""
        st.subheader(" Attack Type Distribution")

        detections = self.load_detections(limit=10000)

        if detections.empty:
            st.info("No detection data available yet.")
            return

        if "attack_type" in detections.columns and "prediction" in detections.columns:
            attacks = detections[
                (detections["prediction"] == 1)
                & (detections["attack_type"].notna())
                & (detections["attack_type"] != "")
                & (detections["attack_type"] != "Benign")
            ]

            if attacks.empty:
                st.info("No attack type classification data available yet.")
                return

            col1, col2 = st.columns(2)

            with col1:
                attack_type_counts = attacks["attack_type"].value_counts()
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=attack_type_counts.index,
                            values=attack_type_counts.values,
                            hole=0.4,
                        )
                    ]
                )
                fig.update_layout(
                    title="Attack Types Detected",
                    height=400,
                    annotations=[
                        dict(text="Types", x=0.5, y=0.5, font_size=20, showarrow=False)
                    ],
                )
                st.plotly_chart(fig, config={'displayModeBar': False})

            with col2:
                top_attacks = attack_type_counts.head(10)
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=top_attacks.values,
                            y=top_attacks.index,
                            orientation="h",
                            marker=dict(color="#e74c3c"),
                        )
                    ]
                )
                fig.update_layout(
                    title="Top 10 Attack Types",
                    xaxis_title="Count",
                    yaxis_title="Attack Type",
                    height=400,
                )
                st.plotly_chart(fig, config={'displayModeBar': False})

            st.subheader(" Attack Type Statistics")

            attack_counts = attacks["attack_type"].value_counts().reset_index()
            attack_counts.columns = ["Attack Type", "Count"]
            st.dataframe(attack_counts, width="stretch")

        else:
            st.info("Attack type classification not enabled or no data available.")

    def render_recent_alerts(self):
        """Render recent alerts table with timestamps and types."""
        st.subheader(" Recent Alerts")

        alerts = self.load_alerts(limit=20)

        if alerts.empty:
            st.success("No alerts generated. System is operating normally.")
            return

        required_cols = [
            "timestamp",
            "packet_id",
            "prediction_label",
            "alert_type",
        ]
        missing_cols = [col for col in required_cols if col not in alerts.columns]

        if missing_cols:
            st.warning(
                f"Alert data incomplete. Missing columns: {', '.join(missing_cols)}"
            )
            return

        st.error(f"**{len(alerts)} recent alert(s) detected!**")

        if "attack_type" in alerts.columns:
            display_alerts = alerts[
                [
                    "timestamp",
                    "packet_id",
                    "prediction_label",
                    "alert_type",
                    "attack_type",
                ]
            ].copy()
        else:
            display_alerts = alerts[
                [
                    "timestamp",
                    "packet_id",
                    "prediction_label",
                    "alert_type",
                ]
            ].copy()

        if "timestamp" in display_alerts.columns and not display_alerts.empty:
            display_alerts["timestamp"] = display_alerts["timestamp"].dt.strftime(
                "%H:%M:%S"
            )

        st.dataframe(
            (
                display_alerts.sort_values("timestamp", ascending=False)
                if "timestamp" in display_alerts.columns
                else display_alerts
            ),
            width="stretch",
            hide_index=True,
        )

    def render_retraining_info(self):
        """Render retraining history with events timeline."""
        st.subheader(" Model Retraining History")

        retrain_df = self.load_retraining()

        if retrain_df.empty:
            st.info("No retraining events yet.")
            return

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=retrain_df["timestamp"],
                    y=retrain_df["samples_used"],
                    mode="markers+lines",
                    name="Samples Used",
                    marker=dict(size=10, color="#3498db"),
                    line=dict(color="#3498db", width=2),
                )
            )
            fig.update_layout(
                title="Retraining Events",
                xaxis_title="Time",
                yaxis_title="Samples Used",
                height=300,
            )
            st.plotly_chart(fig, config={'displayModeBar': False})

        with col2:
            st.metric("Total Retrains", len(retrain_df))
            st.metric("Avg Samples/Retrain", f"{retrain_df['samples_used'].mean():.0f}")
            st.metric(
                "Avg Duration", f"{retrain_df['retrain_duration_ms'].mean():.1f} ms"
            )

    def render_recent_detections(self):
        """Render recent detections table with correctness indicators."""
        with st.expander(" Recent Detections (Last 50)", expanded=False):
            detections = self.load_detections(limit=50)

            if detections.empty:
                st.info("No detections yet.")
                return

            required_cols = [
                "timestamp",
                "packet_id",
                "prediction_label",
                "true_label_name",
                "correct",
            ]
            missing_cols = [
                col for col in required_cols if col not in detections.columns
            ]

            if missing_cols:
                st.warning(
                    f"Detection data incomplete. Missing columns: {', '.join(missing_cols)}"
                )
                return

            cols_to_display = [
                "timestamp",
                "packet_id",
                "prediction_label",
                "true_label_name",
                "correct",
            ]
            if "attack_type" in detections.columns:
                cols_to_display.append("attack_type")
            if "true_attack_type" in detections.columns:
                cols_to_display.append("true_attack_type")

            display_df = detections[cols_to_display].copy()

            if (
                "attack_type" in display_df.columns
                and "true_attack_type" in display_df.columns
            ):
                display_df["attack_type_correct"] = display_df.apply(
                    lambda row: (
                        "✅"
                        if (
                            row["correct"]
                            and row["prediction_label"] == "Attack"
                            and pd.notna(row.get("attack_type"))
                            and pd.notna(row.get("true_attack_type"))
                            and row.get("attack_type") == row.get("true_attack_type")
                        )
                        else (
                            "❌"
                            if (
                                row["prediction_label"] == "Attack"
                                and pd.notna(row.get("attack_type"))
                                and pd.notna(row.get("true_attack_type"))
                            )
                            else ""
                        )
                    ),
                    axis=1,
                )

            if "timestamp" in display_df.columns and not display_df.empty:
                display_df["timestamp"] = display_df["timestamp"].dt.strftime(
                    "%H:%M:%S"
                )
            display_df = display_df.rename(columns={"correct": "binary_correct"})
            display_df["binary_correct"] = display_df["binary_correct"].apply(
                lambda x: "✅" if x else "❌"
            )

            st.dataframe(
                (
                    display_df.sort_values("timestamp", ascending=False)
                    if "timestamp" in display_df.columns
                    else display_df
                ),
                width="stretch",
                hide_index=True,
                height=300,
            )


def main():
    """Main dashboard entry point with sidebar and sections."""
    dashboard = Dashboard(logs_dir="logs")

    with st.sidebar:
        st.header(" Settings")
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        refresh_interval = st.slider("Refresh Interval (seconds)", 1, 10, 3)

        st.markdown("---")
        st.markdown("###  Legend")
        st.markdown("-  **Benign**: Normal traffic")
        st.markdown("-  **Attack**: Malicious traffic")
        st.markdown("-  **Alert**: High-confidence attack detected")

        st.markdown("---")
        st.markdown("###  About")
        st.markdown(
            "Real-time intrusion detection system with incremental learning capabilities."
        )

    dashboard.render_header()
    dashboard.render_metrics_overview()

    st.markdown("---")

    dashboard.render_performance_charts()

    st.markdown("---")

    dashboard.render_detection_distribution()

    st.markdown("---")

    dashboard.render_attack_type_distribution()

    st.markdown("---")

    dashboard.render_recent_alerts()

    st.markdown("---")

    dashboard.render_retraining_info()

    st.markdown("---")

    dashboard.render_recent_detections()

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
