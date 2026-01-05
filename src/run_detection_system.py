import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from simulator.packet_stream import PacketStreamSimulator
from detector.intrusion_detector import IntrusionDetector
from metrics.metrics_tracker import MetricsTracker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntrusionDetectionSystem:
    """Orchestrates packet streaming, detection, learning, and metrics."""

    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        eval_data_path: str,
        packets_per_second: float = 100.0,
        retrain_batch_size: int = 100,
        eval_interval: int = 500,
        output_dir: str = "logs",
        attack_classifier_path: str = None,
        preprocessing_artifacts_path: str = None,
    ):
        """Initialize detection system components.

        Args:
            model_path: Path to trained SGD model
            test_data_path: Test dataset path (RAW or processed)
            eval_data_path: Evaluation dataset path (processed)
            packets_per_second: Stream rate
            retrain_batch_size: Samples per retrain
            eval_interval: Packets between evaluations
            output_dir: Logs directory
            attack_classifier_path: Optional multiclass model
            preprocessing_artifacts_path: Optional path to preprocessing artifacts
                                         (enables real-time preprocessing of RAW packets)
        """
        logger.info("=" * 80)
        logger.info("Initializing Real-Time Intrusion Detection System")
        logger.info("=" * 80)

        # Determine mode based on preprocessing artifacts
        if preprocessing_artifacts_path:
            logger.info("Mode: Real-time preprocessing (RAW packets)")
            logger.info(f"Preprocessing artifacts: {preprocessing_artifacts_path}")
        else:
            logger.info("Mode: Pre-processed packets")

        self.simulator = PacketStreamSimulator(
            data_path=test_data_path,
            packets_per_second=packets_per_second,
            shuffle=True,
            preprocessing_artifacts_path=preprocessing_artifacts_path,
        )

        self.metrics_tracker = MetricsTracker(output_dir=output_dir)

        logger.info(f"Loading evaluation data: {eval_data_path}")
        eval_df = pd.read_csv(eval_data_path)
        columns_to_drop = (
            ["Label", "Attack"] if "Attack" in eval_df.columns else ["Label"]
        )
        self.X_eval = eval_df.drop(columns=columns_to_drop).values
        self.y_eval = eval_df["Label"].values
        logger.info(f"Evaluation set: {len(self.y_eval)} samples")

        self.detector = IntrusionDetector(
            model_path=model_path,
            retrain_batch_size=retrain_batch_size,
            alert_threshold=0.7,
            mcc_threshold=0.85,
            retraining_callback=self.metrics_tracker.log_retraining,
            retrain_cooldown=1200,
            online_metrics_window=2000,
            strict_improvement=True,
            validation_data=(self.X_eval, self.y_eval),
            attack_classifier_path=attack_classifier_path,
        )

        self.eval_interval = eval_interval
        self.packets_since_eval = 0
        self.start_time = time.time()

        logger.info("System initialized")
        logger.info("=" * 80)

    def run(self):
        """Main detection loop: stream packets, predict, learn, log."""
        logger.info("Starting real-time detection...")
        logger.info("Press Ctrl+C to stop\n")

        try:
            for packet in self.simulator.stream():
                self._process_packet(packet)

                self.packets_since_eval += 1
                if self.packets_since_eval >= self.eval_interval:
                    self._evaluate_and_log()
                    self.packets_since_eval = 0

                if packet["packet_id"] % 100 == 0:
                    self._log_progress(packet["packet_id"])

        except KeyboardInterrupt:
            logger.info("\n\nStopping (Ctrl+C)")
        finally:
            self._shutdown()

    def _process_packet(self, packet):
        """Predict, log, and add feedback for a single packet."""
        packet_id = packet["packet_id"]
        features = packet["features"]
        true_label = packet["true_label"]
        true_attack_type = packet.get("true_attack_type")

        result = self.detector.predict(features)

        self.detector.add_feedback(features, true_label)
        self.detector.add_ground_truth(result["prediction"], true_label)

        self.metrics_tracker.log_detection(
            packet_id=packet_id,
            prediction=result["prediction"],
            true_label=true_label,
            alert=result["alert"],
            attack_type=result.get("attack_type"),
            true_attack_type=true_attack_type,
        )

        if result["alert"]:
            self.metrics_tracker.log_alert(
                packet_id=packet_id,
                prediction_label=result["prediction_label"],
                attack_type=result.get("attack_type"),
            )

    def _evaluate_and_log(self):
        """Evaluate model and log performance."""
        logger.info("Evaluating...")
        metrics = self.detector.evaluate_performance(self.X_eval, self.y_eval)
        self.metrics_tracker.log_performance(metrics)
        logger.info(f"Accuracy={metrics['accuracy']:.4f}, MCC={metrics['mcc']:.4f}")

    def _log_progress(self, packet_id: int):
        """Log processing statistics."""
        stats = self.detector.get_statistics()
        stream_stats = self.simulator.get_statistics()
        elapsed = time.time() - self.start_time
        pkt_rate = stats["predictions_count"] / elapsed if elapsed > 0 else 0

        logger.info(
            f"Progress: {packet_id:6d} pkts | "
            f"Attacks: {stats['attacks_detected']:5d} ({stats['attack_rate']:.1f}%) | "
            f"Alerts: {stats['alerts_generated']:4d} | "
            f"Retrains: {stats['retrain_count']:3d} "
            f"(promotion count: {stats.get('promotion_count', 0)}, rejected count: {stats.get('rejected_count', 0)}, reset count: {stats.get('reset_count', 0)}) | "
            f"{pkt_rate:.1f} pkt/s | "
            f"Remaining: {stream_stats['remaining_packets']:6d}"
        )

        self.metrics_tracker.update_summary(extra_stats=stats)

    def _shutdown(self):
        """Finalize and print session summary."""
        logger.info("=" * 80)
        logger.info("Shutting down...")
        self._evaluate_and_log()

        stats = self.detector.get_statistics()
        elapsed = time.time() - self.start_time

        self.metrics_tracker.update_summary(
            extra_stats={
                **stats,
                "session_duration_seconds": elapsed,
                "average_packets_per_second": (
                    stats["predictions_count"] / elapsed if elapsed > 0 else 0
                ),
            }
        )

        logger.info("\n" + "=" * 80)
        logger.info("SESSION SUMMARY")
        logger.info(f"Packets: {stats['predictions_count']:,}")
        logger.info(
            f"Attacks: {stats['attacks_detected']:,} ({stats['attack_rate']:.1f}%)"
        )
        logger.info(f"Benign: {stats['benign_detected']:,}")
        logger.info(f"Alerts: {stats['alerts_generated']:,}")
        logger.info(f"Retrains: {stats['retrain_count']}")
        logger.info(
            f"  Promoted: {stats.get('promotion_count', 0)} ({stats.get('promotion_rate', 0):.1f}%)"
        )
        logger.info(f"  Kept: {stats.get('rejected_count', 0)}")
        logger.info(f"  Reset: {stats.get('reset_count', 0)}")

        if "online_accuracy" in stats:
            logger.info(f"Online Accuracy: {stats['online_accuracy']:.4f}")
            logger.info(f"Online MCC: {stats['online_mcc']:.4f}")

        logger.info(
            f"\nProcessing Rate: {stats['predictions_count'] / elapsed:.1f} pkt/s"
        )
        logger.info(f"Logs: {self.metrics_tracker.output_dir}")
        logger.info("=" * 80)
        logger.info("\nDashboard: streamlit run src/dashboard/streamlit_dashboard.py")


def main():
    """Parse arguments and run system."""
    parser = argparse.ArgumentParser(
        description="Real-Time Intrusion Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python run_detection_system.py
            python run_detection_system.py --rate 500
            python run_detection_system.py --model models/sgd_model.pkl

            Dashboard:
            streamlit run src/dashboard/streamlit_dashboard.py
        """,
    )

    parser.add_argument(
        "--model",
        default="./models/sgd_model_binary.pkl",
        help="Model path (default: models/sgd_model_binary.pkl)",
    )
    parser.add_argument(
        "--test",
        default="./data/processed/test_raw.csv",
        help="Test data path (RAW packets recommended for real-time preprocessing)",
    )
    parser.add_argument(
        "--eval",
        default="./data/processed/eval_data.csv",
        help="Eval data path (must be processed)",
    )
    parser.add_argument(
        "--preprocessing-artifacts",
        default="./data/preprocessing_artifacts",
        help="Path to preprocessing artifacts (enables real-time preprocessing)",
    )
    parser.add_argument("--rate", type=float, default=100.0, help="Packets/sec")
    parser.add_argument(
        "--retrain-batch", type=int, default=1000, help="Retrain batch size"
    )
    parser.add_argument("--eval-interval", type=int, default=500, help="Eval interval")
    parser.add_argument("--output", default="./src/logs", help="Output directory")
    parser.add_argument("--clear-logs", action="store_true", help="Clear logs first")
    parser.add_argument(
        "--attack-classifier",
        default="./models/random_forest_multiclass.pkl",
        help="Attack classifier path",
    )
    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="Disable real-time preprocessing (use pre-processed test data)",
    )

    args = parser.parse_args()

    for path, name in [
        (args.model, "Model"),
        (args.test, "Test data"),
        (args.eval, "Eval data"),
    ]:
        if not Path(path).exists():
            logger.error(f"{name} not found: {path}")
            logger.error("Run data_analysis.ipynb or model_training.ipynb first")
            sys.exit(1)

    if args.clear_logs:
        logger.info("Clearing logs...")
        MetricsTracker(output_dir=args.output).clear_logs()

    attack_classifier_path = None
    if Path(args.attack_classifier).exists():
        attack_classifier_path = args.attack_classifier
        logger.info(f"Using attack classifier: {attack_classifier_path}")
    else:
        logger.warning(f"Attack classifier not found: {args.attack_classifier}")
        logger.warning("Running without attack type classification")

    # Determine preprocessing mode
    preprocessing_artifacts_path = None
    if not args.no_preprocessing:
        if Path(args.preprocessing_artifacts).exists():
            preprocessing_artifacts_path = args.preprocessing_artifacts
            logger.info(f"Real-time preprocessing enabled")
        else:
            logger.warning(
                f"Preprocessing artifacts not found: {args.preprocessing_artifacts}"
            )
            logger.warning(
                "Running without preprocessing (expecting pre-processed test data)"
            )

    system = IntrusionDetectionSystem(
        model_path=args.model,
        test_data_path=args.test,
        eval_data_path=args.eval,
        packets_per_second=args.rate,
        retrain_batch_size=args.retrain_batch,
        eval_interval=args.eval_interval,
        output_dir=args.output,
        attack_classifier_path=attack_classifier_path,
        preprocessing_artifacts_path=preprocessing_artifacts_path,
    )

    system.run()


if __name__ == "__main__":
    main()
