"""Validate preprocessing pipeline matches notebook exactly."""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing.feature_preprocessor import FeaturePreprocessor

def manual_preprocessing(packet, artifacts_path):
    """Manually apply notebook preprocessing pipeline."""

    # Load artifacts
    scaler = joblib.load(artifacts_path / "scaler.pkl")
    imputer = joblib.load(artifacts_path / "imputer.pkl")
    encoders = joblib.load(artifacts_path / "encoders.pkl")
    config = joblib.load(artifacts_path / "feature_config.pkl")

    numerical_cols = config['numerical_cols']
    categorical_cols = config['categorical_cols']

    # Convert packet to DataFrame
    if isinstance(packet, pd.Series):
        df = pd.DataFrame([packet])
    else:
        df = packet.copy()

    # Remove Label and Attack
    df = df.drop(columns=['Label', 'Attack'], errors='ignore')

    # Separate numerical and categorical
    numerical_df = df[[col for col in numerical_cols if col in df.columns]]
    categorical_df = df[[col for col in categorical_cols if col in df.columns]]

    # Impute numerical
    numerical_imputed = pd.DataFrame(
        imputer.transform(numerical_df),
        columns=numerical_df.columns,
        index=df.index
    )

    # Scale numerical
    numerical_scaled = pd.DataFrame(
        scaler.transform(numerical_imputed),
        columns=numerical_df.columns,
        index=df.index
    )

    # Encode categorical
    categorical_encoded = categorical_df.copy()
    for col in categorical_df.columns:
        if col in encoders:
            encoder = encoders[col]
            categorical_encoded[col] = categorical_df[col].astype(str).apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )

    # Concatenate
    processed_df = pd.concat([numerical_scaled, categorical_encoded], axis=1)

    # Remove low variance
    low_variance_cols = config.get('low_variance_features', [])
    processed_df = processed_df.drop(columns=low_variance_cols, errors='ignore')

    # Align features
    final_features = config['final_features']
    for feature in final_features:
        if feature not in processed_df.columns:
            processed_df[feature] = 0

    processed_df = processed_df[final_features]

    return processed_df.values

def main():
    print("="*80)
    print("PREPROCESSING PIPELINE VALIDATION")
    print("="*80)

    base_path = Path(__file__).parent.parent.parent
    test_raw_path = base_path / "data/processed/test_raw.csv"
    artifacts_path = base_path / "data/preprocessing_artifacts"

    if not test_raw_path.exists() or not artifacts_path.exists():
        print("ERROR: Required files not found. Run data_analysis.ipynb first!")
        return

    print("\n1. Loading test data and preprocessor...")
    test_raw = pd.read_csv(test_raw_path)
    preprocessor = FeaturePreprocessor(str(artifacts_path))

    print(f"   test_raw: {test_raw.shape}")
    print(f"   Preprocessor loaded with {len(preprocessor.get_feature_names())} final features")

    print("\n2. Testing on 10 random samples...")
    sample_indices = np.random.choice(len(test_raw), min(10, len(test_raw)), replace=False)

    all_match = True
    for i, idx in enumerate(sample_indices):
        packet = test_raw.iloc[idx]

        # Manual preprocessing
        manual_result = manual_preprocessing(packet, artifacts_path)

        # FeaturePreprocessor
        preprocessor_result = preprocessor.preprocess(packet)

        # Compare
        diff = np.abs(manual_result[0] - preprocessor_result[0])
        max_diff = diff.max()

        if max_diff < 1e-9:
            status = "✓ PERFECT MATCH"
        elif max_diff < 1e-6:
            status = "✓ Good (tiny numerical error)"
        else:
            status = f"✗ MISMATCH (max diff: {max_diff:.6f})"
            all_match = False

        print(f"   Sample {i+1} (idx {idx}): {status}")

        if max_diff >= 1e-6:
            feature_names = preprocessor.get_feature_names()
            large_diffs = np.where(diff > 1e-3)[0]
            if len(large_diffs) > 0:
                print(f"      Features with differences:")
                for feat_idx in large_diffs[:5]:
                    print(f"        {feature_names[feat_idx]}: manual={manual_result[0][feat_idx]:.6f}, preprocessor={preprocessor_result[0][feat_idx]:.6f}")

    print("\n3. Full test on 1000 samples...")
    sample_indices = np.random.choice(len(test_raw), min(1000, len(test_raw)), replace=False)

    max_diffs = []
    for idx in sample_indices:
        packet = test_raw.iloc[idx]
        manual_result = manual_preprocessing(packet, artifacts_path)
        preprocessor_result = preprocessor.preprocess(packet)

        diff = np.abs(manual_result[0] - preprocessor_result[0])
        max_diffs.append(diff.max())

    overall_max = max(max_diffs)
    overall_mean = np.mean(max_diffs)

    print(f"   Max difference across all samples: {overall_max:.10f}")
    print(f"   Mean max difference: {overall_mean:.10f}")

    if overall_max < 1e-9:
        print(f"   ✓ PERFECT MATCH - Preprocessing is identical!")
    elif overall_max < 1e-6:
        print(f"   ✓ EXCELLENT - Only tiny numerical errors")
    elif overall_max < 1e-3:
        print(f"   ⚠️  GOOD - Small numerical differences")
    else:
        print(f"   ✗ FAILED - Significant differences detected!")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
