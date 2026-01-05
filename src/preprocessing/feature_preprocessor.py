"""Feature preprocessor for network packets simulating real-time processing."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """Preprocesses raw network packet features for model inference."""

    def __init__(self, artifacts_path: str):
        """Load preprocessing artifacts (scaler, imputer, encoders, feature config).
        Args: artifacts_path (str) - Directory with preprocessing artifacts.
        """
        self.artifacts_path = Path(artifacts_path)

        self.leakage_features = [
            'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'FLOW_DURATION_MILLISECONDS',
            'DURATION_IN', 'DURATION_OUT', 'SRC_TO_DST_IAT_MIN', 'SRC_TO_DST_IAT_MAX',
            'SRC_TO_DST_IAT_AVG', 'SRC_TO_DST_IAT_STDDEV', 'DST_TO_SRC_IAT_MIN',
            'DST_TO_SRC_IAT_MAX', 'DST_TO_SRC_IAT_AVG', 'DST_TO_SRC_IAT_STDDEV'
        ]

        self.categorical_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'PROTOCOL', 'L7_PROTO']
        self.low_variance_features = ['FTP_COMMAND_RET_CODE']

        self.scaler: Optional[StandardScaler] = None
        self.imputer: Optional[SimpleImputer] = None
        self.encoders: Dict[str, LabelEncoder] = {}
        self.numerical_cols: List[str] = []
        self.final_features: List[str] = []

        self._load_artifacts()

    def _load_artifacts(self):
        """Load scaler, imputer, encoders, and feature config from disk."""
        logger.info(f"Loading preprocessing artifacts from {self.artifacts_path}")

        scaler_path = self.artifacts_path / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded")
        else:
            logger.warning(f"Scaler not found at {scaler_path}")

        imputer_path = self.artifacts_path / "imputer.pkl"
        if imputer_path.exists():
            self.imputer = joblib.load(imputer_path)
            logger.info("Imputer loaded")
        else:
            logger.warning(f"Imputer not found at {imputer_path}")

        encoders_path = self.artifacts_path / "encoders.pkl"
        if encoders_path.exists():
            self.encoders = joblib.load(encoders_path)
            logger.info(f"Loaded {len(self.encoders)} encoders")
        else:
            logger.warning(f"Encoders not found at {encoders_path}")

        features_path = self.artifacts_path / "feature_config.pkl"
        if features_path.exists():
            config = joblib.load(features_path)
            self.numerical_cols = config.get('numerical_cols', [])
            self.final_features = config.get('final_features', [])
            logger.info(f"Loaded feature configuration: {len(self.final_features)} final features")
        else:
            logger.warning(f"Feature config not found at {features_path}")

    def preprocess(self, packet: Union[Dict, pd.Series, pd.DataFrame]) -> np.ndarray:
        """Apply full preprocessing pipeline to raw packet.
        Args: packet (dict/Series/DataFrame) - Raw packet features.
        Returns: np.ndarray - Preprocessed features ready for model.
        """
        if isinstance(packet, dict):
            df = pd.DataFrame([packet])
        elif isinstance(packet, pd.Series):
            df = pd.DataFrame([packet])
        elif isinstance(packet, pd.DataFrame):
            df = packet.copy()
        else:
            raise ValueError(f"Unsupported input type: {type(packet)}")

        df = self._remove_leakage_features(df)

        numerical_df = df[[col for col in self.numerical_cols if col in df.columns]]
        categorical_df = df[[col for col in self.categorical_cols if col in df.columns]]

        numerical_df = self._impute_numerical(numerical_df)
        numerical_df = self._scale_numerical(numerical_df)
        categorical_df = self._encode_categorical(categorical_df)

        processed_df = pd.concat([numerical_df, categorical_df], axis=1)
        processed_df = self._remove_low_variance_features(processed_df)
        processed_df = self._align_features(processed_df)

        return processed_df.values

    def _remove_leakage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove leakage features and metadata (Label, Attack).
        Args: df (DataFrame). Returns: DataFrame.
        """
        columns_to_drop = self.leakage_features + ['Label', 'Attack']
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        return df.drop(columns=columns_to_drop, errors='ignore')

    def _impute_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing numerical values using median strategy.
        Args: df (DataFrame). Returns: DataFrame.
        """
        if self.imputer is None:
            logger.warning("No imputer available, skipping imputation")
            return df

        if df.empty or len(df.columns) == 0:
            return df

        imputed = self.imputer.transform(df)
        return pd.DataFrame(imputed, columns=df.columns, index=df.index)

    def _scale_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features (z-score normalization).
        Args: df (DataFrame). Returns: DataFrame.
        """
        if self.scaler is None:
            logger.warning("No scaler available, skipping scaling")
            return df

        if df.empty or len(df.columns) == 0:
            return df

        scaled = self.scaler.transform(df)
        return pd.DataFrame(scaled, columns=df.columns, index=df.index)

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder, unseen values = -1.
        Args: df (DataFrame). Returns: DataFrame.
        """
        if not self.encoders:
            logger.warning("No encoders available, skipping encoding")
            return df

        encoded_df = df.copy()

        for col in df.columns:
            if col in self.encoders:
                encoder = self.encoders[col]
                # Handle unseen categories
                encoded_df[col] = df[col].astype(str).apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
            else:
                logger.warning(f"No encoder for {col}")

        return encoded_df

    def _remove_low_variance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove low variance features.
        Args: df (DataFrame). Returns: DataFrame.
        """
        columns_to_drop = [col for col in self.low_variance_features if col in df.columns]
        return df.drop(columns=columns_to_drop, errors='ignore')

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct feature order, add missing features as zeros.
        Args: df (DataFrame). Returns: DataFrame.
        """
        if not self.final_features:
            logger.warning("No final feature list available, returning as-is")
            return df

        for feature in self.final_features:
            if feature not in df.columns:
                logger.warning(f"Missing feature {feature}, adding with zeros")
                df[feature] = 0

        return df[self.final_features]

    def preprocess_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess batch of raw packets.
        Args: df (DataFrame) - Raw packets. Returns: np.ndarray - Preprocessed features.
        """
        return self.preprocess(df)

    def get_feature_names(self) -> List[str]:
        """Get final feature names after preprocessing.
        Returns: List[str] - Feature names.
        """
        return self.final_features

    def save_artifacts(self, output_path: str):
        """Save scaler, imputer, encoders, and config to disk.
        Args: output_path (str) - Directory to save artifacts.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.scaler:
            joblib.dump(self.scaler, output_path / "scaler.pkl")
            logger.info("Scaler saved")

        if self.imputer:
            joblib.dump(self.imputer, output_path / "imputer.pkl")
            logger.info("Imputer saved")

        if self.encoders:
            joblib.dump(self.encoders, output_path / "encoders.pkl")
            logger.info("Encoders saved")

        config = {
            'numerical_cols': self.numerical_cols,
            'final_features': self.final_features,
            'categorical_cols': self.categorical_cols,
            'leakage_features': self.leakage_features,
            'low_variance_features': self.low_variance_features
        }
        joblib.dump(config, output_path / "feature_config.pkl")
        logger.info("Feature configuration saved")

        logger.info(f"All artifacts saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python feature_preprocessor.py <artifacts_path>")
        sys.exit(1)

    artifacts_path = sys.argv[1]
    preprocessor = FeaturePreprocessor(artifacts_path)

    # Create dummy raw packet
    dummy_packet = {
        'IPV4_SRC_ADDR': '192.168.1.1',
        'L4_SRC_PORT': 8080,
        'IPV4_DST_ADDR': '10.0.0.1',
        'L4_DST_PORT': 443,
        'PROTOCOL': 6,
        'L7_PROTO': 91,
        'IN_BYTES': 1500,
        'IN_PKTS': 10,
        # ... other features
    }

    try:
        processed = preprocessor.preprocess(dummy_packet)
        print(f"Preprocessed shape: {processed.shape}")
        print(f"Features: {preprocessor.get_feature_names()}")
    except Exception as e:
        print(f"Error: {e}")
