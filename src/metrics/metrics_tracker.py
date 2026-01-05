import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks detections, performance, alerts, and retraining events."""

    def __init__(self, output_dir: str = "logs"):
        """Initialize tracker and create output directory.

        Args:
            output_dir: Directory for CSV/JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.detections_file = self.output_dir / "detections.csv"
        self.performance_file = self.output_dir / "performance.csv"
        self.alerts_file = self.output_dir / "alerts.csv"
        self.retraining_file = self.output_dir / "retraining.csv"
        self.summary_file = self.output_dir / "summary.json"

        self._init_csv_files()

        self.session_start = datetime.now()
        self.total_packets = 0
        self.total_attacks = 0
        self.total_benign = 0
        self.total_alerts = 0
        self.total_retrains = 0

        logger.info(f"Metrics tracker initialized: {output_dir}")

    def _init_csv_files(self):
        """Create CSV files with headers if needed."""

        def _ensure_header(filepath, header):
            if not filepath.exists():
                with open(filepath, "w", newline="") as f:
                    csv.writer(f).writerow(header)
            else:
                with open(filepath, "r", newline="") as f:
                    first_line = f.readline().strip()
                    if first_line and not first_line.startswith(header[0]):
                        f.seek(0)
                        existing_data = f.read()
                        with open(filepath, "w", newline="") as fw:
                            csv.writer(fw).writerow(header)
                            fw.write(existing_data)

        _ensure_header(
            self.detections_file,
            [
                "timestamp",
                "packet_id",
                "prediction",
                "prediction_label",
                "true_label",
                "true_label_name",
                "correct",
                "alert",
                "attack_type",
                "true_attack_type",
            ],
        )

        _ensure_header(
            self.performance_file,
            [
                "timestamp",
                "accuracy",
                "mcc",
                "precision",
                "recall",
                "f1_score",
                "retrain_count",
                "packets_processed",
            ],
        )

        _ensure_header(
            self.alerts_file,
            [
                "timestamp",
                "packet_id",
                "prediction_label",
                "alert_type",
                "attack_type",
            ],
        )

        _ensure_header(
            self.retraining_file,
            [
                "timestamp",
                "retrain_number",
                "samples_used",
                "retrain_duration_ms",
                "model_saved",
            ],
        )

    def log_detection(
        self,
        packet_id: int,
        prediction: int,
        true_label: int,
        alert: bool,
        attack_type: str = None,
        true_attack_type: str = None,
    ):
        """Log detection event to CSV.

        Args:
            packet_id: Packet ID
            prediction: 0=Benign, 1=Attack
            true_label: Ground truth
            alert: Alert triggered
            attack_type: Classified attack type
            true_attack_type: Ground truth attack type
        """
        prediction_label = "Attack" if prediction == 1 else "Benign"
        true_label_name = "Attack" if true_label == 1 else "Benign"

        self.total_packets += 1
        self.total_attacks += prediction == 1
        self.total_benign += prediction == 0
        self.total_alerts += alert

        with open(self.detections_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    datetime.now().isoformat(),
                    packet_id,
                    prediction,
                    prediction_label,
                    true_label,
                    true_label_name,
                    prediction == true_label,
                    alert,
                    attack_type or "",
                    true_attack_type or "",
                ]
            )

    def log_performance(self, metrics: Dict):
        """Log performance metrics to CSV."""
        timestamp = metrics.get("timestamp", datetime.now()).isoformat()

        with open(self.performance_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    timestamp,
                    metrics.get("accuracy", 0),
                    metrics.get("mcc", 0),
                    metrics.get("precision", 0),
                    metrics.get("recall", 0),
                    metrics.get("f1_score", 0),
                    metrics.get("retrain_count", 0),
                    self.total_packets,
                ]
            )

    def log_alert(
        self,
        packet_id: int,
        prediction_label: str,
        alert_type: str = "attack_detected",
        attack_type: str = None,
    ):
        """Log alert event to CSV."""
        with open(self.alerts_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    datetime.now().isoformat(),
                    packet_id,
                    prediction_label,
                    alert_type,
                    attack_type or "",
                ]
            )

    def log_retraining(
        self,
        retrain_number: int,
        samples_used: int,
        duration_ms: float,
        model_saved: bool = False,
    ):
        """Log retraining event to CSV."""
        self.total_retrains += 1

        with open(self.retraining_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    datetime.now().isoformat(),
                    retrain_number,
                    samples_used,
                    duration_ms,
                    model_saved,
                ]
            )

    def update_summary(self, extra_stats: Optional[Dict] = None):
        """Write summary JSON with current statistics."""
        summary = {
            "session_start": self.session_start.isoformat(),
            "last_update": datetime.now().isoformat(),
            "total_packets": self.total_packets,
            "total_attacks_detected": self.total_attacks,
            "total_benign_detected": self.total_benign,
            "total_alerts": self.total_alerts,
            "total_retrains": self.total_retrains,
            "attack_rate_percent": (
                (self.total_attacks / self.total_packets * 100)
                if self.total_packets > 0
                else 0
            ),
        }

        if extra_stats:
            summary.update(extra_stats)

        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)

    def get_summary(self) -> Dict:
        """Read summary JSON."""
        try:
            with open(self.summary_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def clear_logs(self):
        """Delete all log files and reset counters."""
        for file in [
            self.detections_file,
            self.performance_file,
            self.alerts_file,
            self.retraining_file,
            self.summary_file,
        ]:
            if file.exists():
                file.unlink()

        self._init_csv_files()
        self.total_packets = 0
        self.total_attacks = 0
        self.total_benign = 0
        self.total_alerts = 0
        self.total_retrains = 0
        self.session_start = datetime.now()

        logger.info("Logs cleared")


if __name__ == "__main__":
    tracker = MetricsTracker(output_dir="logs")

    print("Testing Metrics Tracker...")
    print("=" * 60)

    for i in range(10):
        tracker.log_detection(
            packet_id=i,
            prediction=i % 2,
            true_label=i % 2,
            alert=(i % 2 == 1),
        )

    tracker.log_performance(
        {
            "accuracy": 0.95,
            "mcc": 0.90,
            "precision": 0.93,
            "recall": 0.92,
            "f1_score": 0.925,
            "retrain_count": 1,
        }
    )

    tracker.log_retraining(retrain_number=1, samples_used=100, duration_ms=45.5)

    tracker.update_summary()

    print("\nSummary:")
    print("=" * 60)
    for key, value in tracker.get_summary().items():
        print(f"{key}: {value}")

    print(f"\nLog files in: {tracker.output_dir}")
    print(f"  Detections: {tracker.detections_file}")
    print(f"  Performance: {tracker.performance_file}")
    print(f"  Alerts: {tracker.alerts_file}")
    print(f"  Retraining: {tracker.retraining_file}")
    print(f"  Summary: {tracker.summary_file}")
