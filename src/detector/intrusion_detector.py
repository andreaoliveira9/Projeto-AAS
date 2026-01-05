"""Real-time intrusion detection with incremental learning capabilities."""

import joblib
import numpy as np
import random
import sys
import time
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from detector.attack_classifier import AttackClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntrusionDetector:
    """Real-time intrusion detector with incremental learning and shadow model architecture."""

    def __init__(
        self,
        model_path: str,
        retrain_batch_size: int = 1000,
        alert_threshold: float = 0.7,
        retrain_interval: int = 1000,
        mcc_threshold: float = 0.60,
        min_retrain_samples: int = 500,
        retraining_callback=None,
        retrain_cooldown: int = 1000,
        online_metrics_window: int = 2000,
        strict_improvement: bool = True,
        validation_data: Tuple[np.ndarray, np.ndarray] = None,
        attack_classifier_path: Optional[str] = None,
    ):
        """Initialize detector with production and candidate models.

        Args:
            model_path: Path to trained SGD model
            retrain_batch_size: Max samples before retraining consideration
            alert_threshold: Confidence threshold for alerts
            retrain_interval: Predictions between automatic retraining
            mcc_threshold: Retrain if MCC falls below this
            min_retrain_samples: Minimum samples for retraining
            retraining_callback: Optional callback for retrain events
            retrain_cooldown: Minimum packets between retrains
            online_metrics_window: Window size for metrics
            strict_improvement: Only promote if MCC improves
            validation_data: (X_val, y_val) for validation
            attack_classifier_path: Path to attack type classifier
        """
        self.model_path = model_path
        self.retrain_batch_size = retrain_batch_size
        self.alert_threshold = alert_threshold
        self.retrain_interval = retrain_interval
        self.mcc_threshold = mcc_threshold
        self.min_retrain_samples = min_retrain_samples
        self.retraining_callback = retraining_callback
        self.retrain_cooldown = retrain_cooldown
        self.online_metrics_window = online_metrics_window
        self.strict_improvement = strict_improvement
        self.validation_data = validation_data

        logger.info(f"Loading model from {model_path}")
        self.production_model = joblib.load(model_path)
        self.candidate_model = None
        logger.info("Production model loaded")

        self.attack_classifier = None
        self.scaler = None
        if attack_classifier_path:
            self._load_attack_classifier(attack_classifier_path, model_path)

        self.best_mcc = None
        if validation_data is not None:
            self._initialize_baseline_mcc(validation_data)

        self.retrain_buffer_benign = []
        self.retrain_buffer_attack = []
        self.retrain_buffer_X = []
        self.retrain_buffer_y = []

        self._initialize_statistics()

        self.performance_history = []
        self.ground_truth_buffer = []
        self.predictions_buffer = []

        logger.info(
            f"Detector initialized - MCC threshold: {self.mcc_threshold}, cooldown: {self.retrain_cooldown}"
        )

    def _load_attack_classifier(self, attack_classifier_path: str, model_path: str):
        """Load attack type classifier and scaler if available."""
        try:
            self.attack_classifier = AttackClassifier(model_path=attack_classifier_path)
            logger.info("Attack type classifier enabled")

            scaler_path = Path(model_path).parent / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")
            else:
                logger.warning(f"Scaler not found at {scaler_path}, using raw data")
        except Exception as e:
            logger.warning(f"Failed to load attack classifier: {e}")

    def _initialize_baseline_mcc(self, validation_data: Tuple[np.ndarray, np.ndarray]):
        """Calculate initial MCC on validation set."""
        X_val, y_val = validation_data
        y_pred = self.production_model.predict(X_val)
        self.best_mcc = matthews_corrcoef(y_val, y_pred)
        logger.info(f"Initial production model MCC: {self.best_mcc:.4f}")

    def _initialize_statistics(self):
        """Initialize all statistics counters."""
        self.predictions_count = 0
        self.attacks_detected = 0
        self.benign_detected = 0
        self.retrain_count = 0
        self.promotion_count = 0
        self.rejected_count = 0
        self.reset_count = 0
        self.alerts_generated = 0
        self.last_retrain_packet = 0
        self.attack_types_detected = {}

    def predict(self, features: Dict) -> Dict:
        """Predict attack/benign using production model and classify attack type if needed.

        Args:
            features: Packet features (raw/unnormalized)

        Returns:
            Prediction results with attack type if applicable
        """
        X_raw = np.array(list(features.values())).reshape(1, -1)
        X_binary = self.scaler.transform(X_raw) if self.scaler else X_raw

        prediction = int(self.production_model.predict(X_binary)[0])
        confidence = self._get_confidence(X_binary, prediction)

        self._update_prediction_stats(prediction)

        alert = self._should_alert(prediction, confidence)
        if alert:
            self.alerts_generated += 1

        result = {
            "prediction": prediction,
            "prediction_label": "Attack" if prediction == 1 else "Benign",
            "confidence": confidence,
            "alert": alert,
            "timestamp": datetime.now(),
            "predictions_count": self.predictions_count,
            "attack_type": None,
            "attack_type_confidence": None,
            "attack_type_probabilities": None,
        }

        if prediction == 1 and self.attack_classifier:
            self._classify_attack_type(features, result)

        return result

    def _get_confidence(self, X: np.ndarray, prediction: int) -> float:
        """Get prediction confidence if available."""
        try:
            proba = self.production_model.predict_proba(X)[0]
            return float(proba[prediction])
        except AttributeError:
            return 1.0

    def _update_prediction_stats(self, prediction: int):
        """Update prediction statistics."""
        self.predictions_count += 1
        if prediction == 1:
            self.attacks_detected += 1
        else:
            self.benign_detected += 1

    def _classify_attack_type(self, features: Dict, result: Dict):
        """Classify attack type and update result dictionary."""
        try:
            attack_result = self.attack_classifier.predict_attack_type(features)
            attack_type = attack_result["attack_type"]

            if attack_type != "Benign":
                self._set_attack_type_result(result, attack_type, attack_result)
            else:
                self._handle_benign_conflict(result, attack_result)
        except Exception as e:
            logger.warning(f"Error classifying attack type: {e}")
            result["attack_type"] = "Unknown"
            result["attack_type_confidence"] = 0.0

    def _set_attack_type_result(
        self, result: Dict, attack_type: str, attack_result: Dict
    ):
        """Set attack type information in result."""
        result["attack_type"] = attack_type
        result["attack_type_confidence"] = attack_result["confidence"]
        result["attack_type_probabilities"] = attack_result["probabilities"]

        if attack_type not in self.attack_types_detected:
            self.attack_types_detected[attack_type] = 0
        self.attack_types_detected[attack_type] += 1

    def _handle_benign_conflict(self, result: Dict, attack_result: Dict):
        """Handle conflict where multiclass predicts Benign but binary predicts Attack."""
        non_benign_probs = {
            k: v for k, v in attack_result["probabilities"].items() if k != "Benign"
        }

        if non_benign_probs:
            best_attack_type = max(non_benign_probs, key=non_benign_probs.get)
            result["attack_type"] = best_attack_type
            result["attack_type_confidence"] = non_benign_probs[best_attack_type]
            result["attack_type_probabilities"] = non_benign_probs

            if best_attack_type not in self.attack_types_detected:
                self.attack_types_detected[best_attack_type] = 0
            self.attack_types_detected[best_attack_type] += 1
        else:
            result["attack_type"] = "Unknown"
            result["attack_type_confidence"] = 0.0

    def _should_alert(self, prediction: int, confidence: float) -> bool:
        """Check if alert should be generated for attack prediction."""
        return prediction == 1 and confidence >= self.alert_threshold

    def _should_retrain(self) -> bool:
        """Determine if model should be retrained based on performance and data quality."""
        if not self._has_sufficient_data():
            return False

        if not self._check_cooldown():
            return False

        current_metrics = self.get_online_metrics()
        if current_metrics is None:
            logger.debug("No online metrics available")
            return False

        current_mcc = current_metrics["mcc"]

        if self._check_mcc_degradation(current_mcc):
            return True

        if self._check_concept_drift(current_mcc):
            return True

        return False

    def _has_sufficient_data(self) -> bool:
        """Check if we have enough balanced data for retraining."""
        n_benign = len(self.retrain_buffer_benign)
        n_attack = len(self.retrain_buffer_attack)
        total_samples = n_benign + n_attack

        if total_samples < self.min_retrain_samples:
            logger.debug(
                f"Insufficient samples: {total_samples} < {self.min_retrain_samples}"
            )
            return False

        if n_benign == 0 or n_attack == 0:
            logger.debug(f"Missing class: Benign={n_benign}, Attack={n_attack}")
            return False

        minority_ratio = min(n_benign, n_attack) / total_samples
        if minority_ratio < 0.20:
            logger.warning(
                f"Imbalanced data: Benign={n_benign} ({n_benign/total_samples:.1%}), "
                f"Attack={n_attack} ({n_attack/total_samples:.1%}). Need 20% minimum."
            )
            return False

        return True

    def _check_cooldown(self) -> bool:
        """Check if cooldown period has elapsed."""
        packets_since_retrain = self.predictions_count - self.last_retrain_packet
        if packets_since_retrain < self.retrain_cooldown:
            logger.debug(
                f"Cooldown active: {packets_since_retrain} < {self.retrain_cooldown}"
            )
            return False
        return True

    def _check_mcc_degradation(self, current_mcc: float) -> bool:
        """Check if MCC has degraded below threshold."""
        if current_mcc < self.mcc_threshold:
            if current_mcc < 0.30:
                logger.error(
                    f"MCC critically low ({current_mcc:.4f}), possible corruption"
                )
                return False
            logger.warning(f"MCC degraded: {current_mcc:.4f} < {self.mcc_threshold}")
            return True
        return False

    def _check_concept_drift(self, current_mcc: float) -> bool:
        """Check for concept drift through error rate analysis."""
        if len(self.predictions_buffer) < 150:
            return False

        recent_errors = np.sum(
            np.array(self.predictions_buffer[-75:])
            != np.array(self.ground_truth_buffer[-75:])
        )
        older_errors = np.sum(
            np.array(self.predictions_buffer[-150:-75])
            != np.array(self.ground_truth_buffer[-150:-75])
        )

        if recent_errors > older_errors * 1.60 and recent_errors > 15:
            if current_mcc >= self.mcc_threshold:
                logger.info(
                    f"Drift detected but MCC good ({current_mcc:.4f}), skipping"
                )
                return False

            recent_rate = recent_errors / 75.0
            older_rate = older_errors / 75.0
            logger.warning(
                f"Drift + degraded MCC: {recent_rate:.2%} vs {older_rate:.2%}"
            )
            return True

        return False

    def add_feedback(self, features: Dict, true_label: int):
        """Add labeled sample to stratified retrain buffer.

        Args:
            features: Packet features
            true_label: Ground truth (0=Benign, 1=Attack)
        """
        feature_values = list(features.values())

        if true_label == 0:
            self.retrain_buffer_benign.append(feature_values)
        else:
            self.retrain_buffer_attack.append(feature_values)

        self.retrain_buffer_X.append(feature_values)
        self.retrain_buffer_y.append(true_label)

        total_buffer_size = len(self.retrain_buffer_benign) + len(
            self.retrain_buffer_attack
        )

        if total_buffer_size >= self.retrain_batch_size:
            if (
                len(self.retrain_buffer_benign) > 0
                and len(self.retrain_buffer_attack) > 0
            ):
                current_metrics = self.get_online_metrics()

                if current_metrics and current_metrics["mcc"] < 0.25:
                    logger.error(
                        f"Emergency: MCC too low ({current_metrics['mcc']:.4f}), clearing buffer"
                    )
                    self._clear_buffers()
                    return

                if self._should_retrain():
                    self.retrain(metrics_callback=self.retraining_callback)
                else:
                    self._trim_buffers()
            else:
                logger.debug(
                    f"Waiting for both classes: Benign={len(self.retrain_buffer_benign)}, Attack={len(self.retrain_buffer_attack)}"
                )

    def _clear_buffers(self):
        """Clear all retrain buffers."""
        self.retrain_buffer_benign = []
        self.retrain_buffer_attack = []
        self.retrain_buffer_X = []
        self.retrain_buffer_y = []

    def _trim_buffers(self):
        """Trim buffers to prevent memory issues."""
        max_size = self.retrain_batch_size
        if len(self.retrain_buffer_benign) > max_size:
            self.retrain_buffer_benign = self.retrain_buffer_benign[-max_size:]
        if len(self.retrain_buffer_attack) > max_size:
            self.retrain_buffer_attack = self.retrain_buffer_attack[-max_size:]

    def retrain(self, force: bool = False, metrics_callback=None):
        """Train candidate model with stratified balanced samples using shadow model strategy.

        Args:
            force: Bypass retrain decision logic
            metrics_callback: Optional callback for retrain metrics
        """
        n_benign = len(self.retrain_buffer_benign)
        n_attack = len(self.retrain_buffer_attack)

        if n_benign == 0 or n_attack == 0:
            logger.warning(
                f"Cannot retrain: need both classes (Benign={n_benign}, Attack={n_attack})"
            )
            return

        if not force and not self._should_start_retrain():
            return None

        X_new, y_new, samples_used = self._prepare_training_data(n_benign, n_attack)
        if X_new is None:
            return

        self._initialize_candidate_model()

        retrain_time = self._train_candidate(X_new, y_new, samples_used, force)

        promoted, should_reset = self._evaluate_and_promote_candidate()
        self._handle_candidate_status(promoted, should_reset)

        self._finalize_retrain(samples_used, retrain_time, promoted, metrics_callback)

        return retrain_time

    def _should_start_retrain(self) -> bool:
        """Check if we should start retraining based on current performance."""
        previous_metrics = self.get_online_metrics()
        production_mcc = self.best_mcc if self.best_mcc is not None else 0.0
        logger.info(
            f"Starting candidate training. Production MCC: {production_mcc:.4f}"
        )

        if previous_metrics and previous_metrics["mcc"] >= 0.85:
            logger.info(
                f"Production model excellent (MCC={previous_metrics['mcc']:.4f}), skipping"
            )
            return False
        return True

    def _prepare_training_data(self, n_benign: int, n_attack: int) -> Tuple:
        """Prepare balanced training data from buffers."""
        min_class_size = min(n_benign, n_attack)
        max_samples_per_class = min(min_class_size, self.retrain_batch_size // 2)

        if max_samples_per_class < 50:
            logger.warning(
                f"Insufficient samples: {max_samples_per_class} < 50 per class"
            )
            return None, None, None

        benign_samples = random.sample(
            self.retrain_buffer_benign, max_samples_per_class
        )
        attack_samples = random.sample(
            self.retrain_buffer_attack, max_samples_per_class
        )

        X_new = np.array(benign_samples + attack_samples)
        y_new = np.array([0] * len(benign_samples) + [1] * len(attack_samples))

        shuffle_idx = np.random.permutation(len(X_new))
        X_new = X_new[shuffle_idx]
        y_new = y_new[shuffle_idx]

        samples_used = len(X_new)
        balance_ratio = (
            len(benign_samples) / len(attack_samples)
            if len(attack_samples) > 0
            else 1.0
        )

        logger.info(
            f"Training with {samples_used} balanced samples "
            f"(Benign={len(benign_samples)}, Attack={len(attack_samples)}, Ratio={balance_ratio:.2f}:1)"
        )

        return X_new, y_new, samples_used

    def _initialize_candidate_model(self):
        """Create or continue candidate model."""
        import copy

        if self.candidate_model is None:
            self.candidate_model = copy.deepcopy(self.production_model)
            logger.info("Candidate created from production")
        else:
            logger.info("Continuing candidate training")

    def _train_candidate(
        self, X_new: np.ndarray, y_new: np.ndarray, samples_used: int, force: bool
    ) -> float:
        """Train candidate model in mini-batches."""
        start_time = time.time()
        mini_batch_size = 100

        for i in range(0, len(X_new), mini_batch_size):
            X_batch = X_new[i : i + mini_batch_size]
            y_batch = y_new[i : i + mini_batch_size]
            self.candidate_model.partial_fit(X_batch, y_batch, classes=[0, 1])

        retrain_time = (time.time() - start_time) * 1000
        logger.info(f"Training completed in {retrain_time:.3f}ms")
        return retrain_time

    def _handle_candidate_status(self, promoted: bool, should_reset: bool):
        """Handle candidate model based on evaluation results."""
        import copy

        if promoted:
            logger.info("Creating new candidate from production")
            self.candidate_model = copy.deepcopy(self.production_model)
        elif should_reset:
            logger.warning("Resetting candidate due to corruption")
            self.candidate_model = copy.deepcopy(self.production_model)
        else:
            logger.info("Keeping candidate for next iteration")

    def _finalize_retrain(
        self, samples_used: int, retrain_time: float, promoted: bool, metrics_callback
    ):
        """Update statistics and trim buffers after retraining."""
        self.retrain_count += 1
        self.last_retrain_packet = self.predictions_count

        if metrics_callback:
            metrics_callback(
                retrain_number=self.retrain_count,
                samples_used=samples_used,
                duration_ms=retrain_time,
                model_saved=promoted,
            )

        keep_size_benign = int(len(self.retrain_buffer_benign) * 0.6)
        keep_size_attack = int(len(self.retrain_buffer_attack) * 0.6)

        self.retrain_buffer_benign = self.retrain_buffer_benign[-keep_size_benign:]
        self.retrain_buffer_attack = self.retrain_buffer_attack[-keep_size_attack:]

        self.retrain_buffer_X = self.retrain_buffer_benign + self.retrain_buffer_attack
        self.retrain_buffer_y = [0] * len(self.retrain_buffer_benign) + [1] * len(
            self.retrain_buffer_attack
        )

        logger.debug(
            f"Buffer: Benign={len(self.retrain_buffer_benign)}, Attack={len(self.retrain_buffer_attack)}"
        )

    def _evaluate_and_promote_candidate(self) -> Tuple[bool, bool]:
        """Evaluate candidate and decide promotion or reset.

        Returns:
            (promoted, should_reset) tuple
        """
        if self.candidate_model is None:
            logger.error("No candidate model to evaluate")
            return False, True

        if self.validation_data is None:
            logger.warning("No validation dataset available")
            self.rejected_count += 1
            return False, False

        X_val, y_val = self.validation_data

        try:
            candidate_predictions = self.candidate_model.predict(X_val)
            candidate_mcc = matthews_corrcoef(y_val, candidate_predictions)

            if self._is_corrupted(candidate_predictions, candidate_mcc):
                return False, True

            production_mcc = self.best_mcc if self.best_mcc is not None else 0.0

            if self._has_severe_degradation(candidate_mcc, production_mcc):
                return False, True

            return self._decide_promotion(candidate_mcc, production_mcc)

        except Exception as e:
            logger.error(f"Error evaluating candidate: {e}")
            self.reset_count += 1
            return False, True

    def _is_corrupted(self, predictions: np.ndarray, mcc: float) -> bool:
        """Check if candidate model is corrupted."""
        unique_predictions = np.unique(predictions)
        if len(unique_predictions) == 1:
            logger.error(
                f"Candidate predicts only class {unique_predictions[0]}, resetting"
            )
            self.reset_count += 1
            return True

        if mcc < 0.30:
            logger.error(f"Candidate MCC too low ({mcc:.4f}), resetting")
            self.reset_count += 1
            return True

        return False

    def _has_severe_degradation(
        self, candidate_mcc: float, production_mcc: float
    ) -> bool:
        """Check if candidate has severe degradation compared to production."""
        if production_mcc > 0 and candidate_mcc < production_mcc * 0.80:
            degradation = (production_mcc - candidate_mcc) / production_mcc * 100
            logger.error(
                f"Severe degradation: Candidate {candidate_mcc:.4f} << Production {production_mcc:.4f} (-{degradation:.1f}%)"
            )
            self.reset_count += 1
            return True
        return False

    def _decide_promotion(
        self, candidate_mcc: float, production_mcc: float
    ) -> Tuple[bool, bool]:
        """Decide whether to promote candidate based on improvement mode."""
        if self.strict_improvement:
            return self._evaluate_strict_mode(candidate_mcc, production_mcc)
        else:
            return self._evaluate_tolerant_mode(candidate_mcc, production_mcc)

    def _evaluate_strict_mode(
        self, candidate_mcc: float, production_mcc: float
    ) -> Tuple[bool, bool]:
        """Evaluate candidate in strict improvement mode."""
        if candidate_mcc > production_mcc:
            improvement = (
                ((candidate_mcc - production_mcc) / production_mcc * 100)
                if production_mcc > 0
                else 100
            )
            logger.info(
                f"Candidate promoted: {candidate_mcc:.4f} > {production_mcc:.4f} ({improvement:+.1f}%)"
            )
            self._promote_candidate(candidate_mcc)
            return True, False
        else:
            logger.warning(
                f"Candidate not promoted: {candidate_mcc:.4f} <= {production_mcc:.4f}"
            )
            self.rejected_count += 1
            return False, False

    def _evaluate_tolerant_mode(
        self, candidate_mcc: float, production_mcc: float
    ) -> Tuple[bool, bool]:
        """Evaluate candidate in tolerant mode (allows small degradation)."""
        min_acceptable_mcc = max(production_mcc * 0.95, 0.50)

        if candidate_mcc >= min_acceptable_mcc:
            if candidate_mcc > production_mcc:
                improvement = (
                    ((candidate_mcc - production_mcc) / production_mcc * 100)
                    if production_mcc > 0
                    else 100
                )
                logger.info(
                    f"Candidate promoted: {candidate_mcc:.4f} > {production_mcc:.4f} ({improvement:+.1f}%)"
                )
            else:
                logger.info(
                    f"Candidate promoted (tolerant): {candidate_mcc:.4f} >= {min_acceptable_mcc:.4f}"
                )
            self._promote_candidate(candidate_mcc)
            return True, False
        else:
            logger.warning(
                f"Candidate rejected: {candidate_mcc:.4f} < {min_acceptable_mcc:.4f}"
            )
            self.rejected_count += 1
            return False, False

    def _promote_candidate(self, candidate_mcc: float):
        """Promote candidate to production."""
        self.production_model = self.candidate_model
        self.best_mcc = candidate_mcc
        self.promotion_count += 1
        self._save_production_model()

    def _save_production_model(self):
        """Save current production model to disk."""
        try:
            joblib.dump(self.production_model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def evaluate_performance(self, X_eval: np.ndarray, y_eval: np.ndarray) -> Dict:
        """Evaluate production model performance.

        Args:
            X_eval: Evaluation features
            y_eval: Evaluation labels

        Returns:
            Performance metrics dictionary
        """
        y_pred = self.production_model.predict(X_eval)

        metrics = {
            "accuracy": accuracy_score(y_eval, y_pred),
            "mcc": matthews_corrcoef(y_eval, y_pred),
            "precision": precision_score(y_eval, y_pred, zero_division=0),
            "recall": recall_score(y_eval, y_pred, zero_division=0),
            "f1_score": f1_score(y_eval, y_pred, zero_division=0),
            "timestamp": datetime.now(),
            "retrain_count": self.retrain_count,
        }

        self.performance_history.append(metrics)

        if self.best_mcc is None:
            self.best_mcc = metrics["mcc"]
            logger.info(f"Initial MCC baseline: {self.best_mcc:.4f}")

        logger.info(
            f"Performance: Accuracy={metrics['accuracy']:.4f}, MCC={metrics['mcc']:.4f}"
        )

        return metrics

    def add_ground_truth(self, prediction: int, true_label: int):
        """Track prediction vs ground truth for online monitoring.

        Args:
            prediction: Model prediction
            true_label: True label
        """
        self.predictions_buffer.append(prediction)
        self.ground_truth_buffer.append(true_label)

        if len(self.predictions_buffer) > self.online_metrics_window:
            self.predictions_buffer.pop(0)
            self.ground_truth_buffer.pop(0)

    def get_online_metrics(self) -> Optional[Dict]:
        """Calculate metrics from buffered predictions.

        Returns:
            Metrics dictionary or None if insufficient data
        """
        if len(self.predictions_buffer) < 10:
            return None

        y_true = np.array(self.ground_truth_buffer)
        y_pred = np.array(self.predictions_buffer)

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "buffer_size": len(self.predictions_buffer),
            "timestamp": datetime.now(),
        }

    def get_statistics(self) -> Dict:
        """Get detector statistics.

        Returns:
            Current statistics dictionary
        """
        stats = {
            "predictions_count": self.predictions_count,
            "attacks_detected": self.attacks_detected,
            "benign_detected": self.benign_detected,
            "attack_rate": (
                (self.attacks_detected / self.predictions_count * 100)
                if self.predictions_count > 0
                else 0
            ),
            "retrain_count": self.retrain_count,
            "promotion_count": self.promotion_count,
            "rejected_count": self.rejected_count,
            "reset_count": self.reset_count,
            "promotion_rate": (
                (self.promotion_count / self.retrain_count * 100)
                if self.retrain_count > 0
                else 0
            ),
            "alerts_generated": self.alerts_generated,
            "retrain_buffer_size": len(self.retrain_buffer_X),
            "performance_history_size": len(self.performance_history),
            "has_candidate": self.candidate_model is not None,
            "attack_types_detected": dict(self.attack_types_detected),
            "attack_classifier_enabled": self.attack_classifier is not None,
        }

        online_metrics = self.get_online_metrics()
        if online_metrics:
            stats["online_accuracy"] = online_metrics["accuracy"]
            stats["online_mcc"] = online_metrics["mcc"]

        return stats

    def save_model(self, path: str = None, force: bool = False):
        """Save production model to disk.

        Args:
            path: Save path (default: self.model_path)
            force: Save even without validation

        Returns:
            True if successful
        """
        save_path = path or self.model_path

        try:
            joblib.dump(self.production_model, save_path)
            logger.info(f"Model saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python intrusion_detector.py <path_to_model.pkl>")
        sys.exit(1)

    model_path = sys.argv[1]

    # Test detector
    detector = IntrusionDetector(model_path=model_path)

    # Create dummy features (39 features as per training)
    dummy_features = {f"feature_{i}": np.random.randn() for i in range(39)}

    # Make prediction
    result = detector.predict(dummy_features)

    print("\nPrediction Result:")
    print("=" * 60)
    for key, value in result.items():
        print(f"  {key}: {value}")

    print("\nDetector Statistics:")
    print("=" * 60)
    stats = detector.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
