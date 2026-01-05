import joblib
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class AttackClassifier:
    """Classifies specific attack types when attacks are detected."""

    def __init__(self, model_path: str):
        """Load attack classifier model.

        Args:
            model_path: Path to trained multiclass model (.pkl)
        """
        self.model_path = model_path
        logger.info(f"Loading attack classifier: {model_path}")

        self.model = joblib.load(model_path)
        self.attack_types = (
            list(self.model.classes_) if hasattr(self.model, "classes_") else []
        )

        if self.attack_types:
            logger.info(f"Loaded {len(self.attack_types)} attack types")
        else:
            logger.warning("Model missing classes_ attribute")

    def predict_attack_type(self, features: Dict) -> Dict:
        """Predict attack type and confidence.

        Args:
            features: Packet feature dictionary

        Returns:
            Dict with attack_type, confidence, and top-3 probabilities
        """
        X = np.array(list(features.values())).reshape(1, -1)
        prediction = self.model.predict(X)[0]

        attack_type, attack_type_idx = self._get_attack_info(prediction)
        confidence, probabilities = self._get_probabilities(
            X, attack_type_idx, attack_type
        )

        return {
            "attack_type": attack_type,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def _get_attack_info(self, prediction):
        """Extract attack type and index from prediction."""
        if isinstance(prediction, str):
            attack_type = prediction
            try:
                attack_type_idx = (
                    self.attack_types.index(prediction) if self.attack_types else 0
                )
            except ValueError:
                attack_type_idx = 0
        else:
            attack_type_idx = int(prediction)
            attack_type = (
                self.attack_types[attack_type_idx]
                if self.attack_types
                else f"Attack_{attack_type_idx}"
            )
        return attack_type, attack_type_idx

    def _get_probabilities(self, X, attack_type_idx, attack_type):
        """Get confidence and top-3 attack probabilities."""
        try:
            proba = self.model.predict_proba(X)[0]
            confidence = float(proba[attack_type_idx])

            top_indices = np.argsort(proba)[-3:][::-1]
            probabilities = {
                (
                    self.attack_types[idx] if self.attack_types else f"Attack_{idx}"
                ): float(proba[idx])
                for idx in top_indices
            }
        except AttributeError:
            confidence = 1.0
            probabilities = {attack_type: 1.0}

        return confidence, probabilities

    def get_statistics(self) -> Dict:
        """Get classifier configuration and metadata."""
        return {
            "model_path": self.model_path,
            "num_attack_types": len(self.attack_types),
            "attack_types": self.attack_types,
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python attack_classifier.py <model.pkl>")
        sys.exit(1)

    classifier = AttackClassifier(sys.argv[1])
    dummy_features = {f"feature_{i}": np.random.randn() for i in range(39)}
    result = classifier.predict_attack_type(dummy_features)

    print("\nPrediction Result:")
    print("=" * 60)
    print(f"Attack Type: {result['attack_type']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nTop Probabilities:")
    for attack_type, prob in result["probabilities"].items():
        print(f"  {attack_type}: {prob:.4f}")

    print("\nStatistics:")
    print("=" * 60)
    stats = classifier.get_statistics()
    for key, value in stats.items():
        if key != "attack_types":
            print(f"{key}: {value}")
