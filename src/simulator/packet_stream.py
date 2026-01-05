import pandas as pd
import time
import sys
from pathlib import Path
from typing import Iterator, Dict, Optional
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class PacketStreamSimulator:
    """Streams packets from dataset at configurable rate with optional preprocessing."""

    def __init__(
        self,
        data_path: str,
        packets_per_second: float = 100.0,
        shuffle: bool = True,
        random_state: int = 42,
        preprocessing_artifacts_path: Optional[str] = None,
    ):
        """Initialize packet stream simulator.
        Args: data_path (str), packets_per_second (float), shuffle (bool),
              random_state (int), preprocessing_artifacts_path (str, optional).
        """
        self.data_path = data_path
        self.packets_per_second = packets_per_second
        self.delay = 1.0 / packets_per_second
        self.shuffle = shuffle
        self.random_state = random_state

        self.preprocessor = None
        if preprocessing_artifacts_path:
            self._load_preprocessor(preprocessing_artifacts_path)

        logger.info(f"Loading dataset: {data_path}")
        self.df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.df)} packets")

        if self.shuffle:
            self.df = self.df.sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)
            logger.info("Dataset shuffled")

        self.current_index = 0
        self.total_packets = len(self.df)

    def _load_preprocessor(self, artifacts_path: str):
        """Load FeaturePreprocessor from artifacts.
        Args: artifacts_path (str).
        """
        try:
            from preprocessing.feature_preprocessor import FeaturePreprocessor
            self.preprocessor = FeaturePreprocessor(artifacts_path)
            logger.info(f"Preprocessor loaded from {artifacts_path}")
            logger.info("Raw packets will be preprocessed in real-time")
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            logger.warning("Streaming without preprocessing")

    def stream(self) -> Iterator[Dict]:
        """Stream packets with timing and optional preprocessing.
        Yields: Dict with packet_id, timestamp, features, true_label, true_attack_type, raw_features (optional).
        """
        mode = "with real-time preprocessing" if self.preprocessor else "without preprocessing"
        logger.info(f"Streaming at {self.packets_per_second} pkt/s ({mode})")

        while self.current_index < self.total_packets:
            packet = self.df.iloc[self.current_index]
            label = int(packet["Label"])

            true_attack_type = self._extract_attack_type(packet, label)
            raw_features = self._extract_features(packet)

            if self.preprocessor:
                try:
                    preprocessed = self.preprocessor.preprocess(raw_features)
                    feature_names = self.preprocessor.get_feature_names()
                    features = {name: float(val) for name, val in zip(feature_names, preprocessed[0])}
                except Exception as e:
                    logger.error(f"Preprocessing error on packet {self.current_index}: {e}")
                    features = raw_features
            else:
                features = raw_features

            result = {
                "packet_id": self.current_index,
                "timestamp": time.time(),
                "features": features,
                "true_label": label,
                "true_attack_type": true_attack_type,
            }

            if self.preprocessor:
                result["raw_features"] = raw_features

            yield result

            self.current_index += 1
            time.sleep(self.delay)

        logger.info("Stream completed")

    def _extract_attack_type(self, packet, label):
        """Extract attack type from packet.
        Args: packet (Series), label (int). Returns: str or None.
        """
        if "Attack" not in packet.index or label == 0:
            return None

        attack_val = packet["Attack"]
        return str(attack_val) if pd.notna(attack_val) and attack_val != "" else None

    def _extract_features(self, packet):
        """Extract features dict from packet, excluding Label and Attack.
        Args: packet (Series). Returns: dict.
        """
        columns_to_drop = ["Label"]
        if "Attack" in packet.index:
            columns_to_drop.append("Attack")
        return packet.drop(columns_to_drop).to_dict()

    def get_batch(self, batch_size: int) -> Optional[pd.DataFrame]:
        """Get batch of packets.
        Args: batch_size (int). Returns: DataFrame or None if exhausted.
        """
        if self.current_index >= self.total_packets:
            return None

        end_index = min(self.current_index + batch_size, self.total_packets)
        batch = self.df.iloc[self.current_index : end_index].copy()
        self.current_index = end_index
        return batch

    def reset(self):
        """Reset stream to beginning."""
        self.current_index = 0
        if self.shuffle:
            self.df = self.df.sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)
        logger.info("Stream reset")

    def get_progress(self) -> float:
        """Get progress percentage. Returns: float."""
        return (self.current_index / self.total_packets) * 100

    def get_remaining(self) -> int:
        """Get remaining packet count. Returns: int."""
        return self.total_packets - self.current_index

    def get_statistics(self) -> Dict:
        """Get stream statistics. Returns: dict."""
        processed = self.df.iloc[: self.current_index]
        remaining = self.df.iloc[self.current_index :]

        stats = {
            "total_packets": self.total_packets,
            "processed_packets": self.current_index,
            "remaining_packets": self.get_remaining(),
            "progress_percent": self.get_progress(),
            "packets_per_second": self.packets_per_second,
        }

        if len(processed) > 0:
            stats.update(
                {
                    "processed_benign": int((processed["Label"] == 0).sum()),
                    "processed_attacks": int((processed["Label"] == 1).sum()),
                }
            )

        if len(remaining) > 0:
            stats.update(
                {
                    "remaining_benign": int((remaining["Label"] == 0).sum()),
                    "remaining_attacks": int((remaining["Label"] == 1).sum()),
                }
            )

        return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python packet_stream.py <test_data.csv>")
        sys.exit(1)

    simulator = PacketStreamSimulator(
        data_path=sys.argv[1], packets_per_second=10, shuffle=True
    )

    print("\nStreaming first 20 packets...\n")
    for i, packet in enumerate(simulator.stream()):
        if i >= 20:
            break
        label = "Attack" if packet["true_label"] == 1 else "Benign"
        print(f"Packet {packet['packet_id']:5d} | {packet['timestamp']:.2f} | {label}")

    print("\n" + "=" * 60)
    print("Statistics:")
    print("=" * 60)
    for key, value in simulator.get_statistics().items():
        print(f"{key}: {value}")
