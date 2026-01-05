"""Script to validate preprocessing consistency."""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing.feature_preprocessor import FeaturePreprocessor

def test_preprocessing_consistency():
    """Compare preprocessor output with pre-processed dataset."""

    print("="*80)
    print("PREPROCESSING VALIDATION TEST")
    print("="*80)

    base_path = Path(__file__).parent.parent.parent

    test_raw_path = base_path / "data/processed/test_raw.csv"
    test_processed_path = base_path / "data/processed/test_data.csv"
    artifacts_path = base_path / "data/preprocessing_artifacts"

    if not test_raw_path.exists():
        print(f"ERROR: test_raw.csv not found at {test_raw_path}")
        print("Run data_analysis.ipynb first!")
        return

    if not test_processed_path.exists():
        print(f"ERROR: test_data.csv not found at {test_processed_path}")
        return

    if not artifacts_path.exists():
        print(f"ERROR: preprocessing_artifacts not found at {artifacts_path}")
        return

    print("\n1. Loading datasets...")
    test_raw = pd.read_csv(test_raw_path)
    test_processed = pd.read_csv(test_processed_path)

    print(f"   test_raw: {test_raw.shape}")
    print(f"   test_processed: {test_processed.shape}")

    print("\n2. Checking test_raw columns...")
    print(f"   Columns in test_raw: {list(test_raw.columns)[:10]}...")

    leakage_features_in_raw = [
        'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
        'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT'
    ]

    found_leakage = [f for f in leakage_features_in_raw if f in test_raw.columns]
    if found_leakage:
        print(f"   ⚠️  WARNING: Leakage features found in test_raw: {found_leakage}")
        print("   test_raw should NOT contain leakage features!")
    else:
        print("   ✓ No leakage features in test_raw (correct)")

    print("\n3. Loading preprocessor...")
    preprocessor = FeaturePreprocessor(str(artifacts_path))

    print("\n4. Processing a sample packet...")
    sample_packet = test_raw.iloc[0]
    sample_id = 0

    print(f"   Sample packet columns: {list(sample_packet.index)[:10]}...")

    try:
        processed_features = preprocessor.preprocess(sample_packet)
        print(f"   ✓ Preprocessing successful")
        print(f"   Output shape: {processed_features.shape}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n5. Comparing with expected output...")
    expected_features = test_processed.iloc[sample_id].drop(['Label', 'Attack']).values

    print(f"   Expected shape: {expected_features.shape}")
    print(f"   Preprocessor shape: {processed_features.shape}")

    if expected_features.shape != processed_features.shape[1:]:
        print(f"   ✗ SHAPE MISMATCH!")
        return

    diff = np.abs(expected_features - processed_features[0])
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"   Max difference: {max_diff:.10f}")
    print(f"   Mean difference: {mean_diff:.10f}")

    if max_diff < 1e-6:
        print(f"   ✓ PERFECT MATCH!")
    elif max_diff < 1e-3:
        print(f"   ✓ Good match (small numerical differences)")
    else:
        print(f"   ⚠️  WARNING: Large differences detected!")

        feature_names = preprocessor.get_feature_names()
        large_diffs = np.where(diff > 1e-3)[0]

        if len(large_diffs) > 0:
            print(f"\n   Features with large differences ({len(large_diffs)}):")
            for idx in large_diffs[:10]:
                print(f"     {feature_names[idx]}: expected={expected_features[idx]:.6f}, got={processed_features[0][idx]:.6f}, diff={diff[idx]:.6f}")

    print("\n6. Testing on 100 random samples...")
    sample_indices = np.random.choice(len(test_raw), min(100, len(test_raw)), replace=False)

    all_diffs = []
    for idx in sample_indices:
        packet = test_raw.iloc[idx]
        processed = preprocessor.preprocess(packet)
        expected = test_processed.iloc[idx].drop(['Label', 'Attack']).values

        diff = np.abs(expected - processed[0])
        all_diffs.append(diff.max())

    max_of_all = max(all_diffs)
    mean_of_all = np.mean(all_diffs)

    print(f"   Max difference across all samples: {max_of_all:.10f}")
    print(f"   Mean max difference: {mean_of_all:.10f}")

    if max_of_all < 1e-6:
        print(f"   ✓ ALL SAMPLES MATCH PERFECTLY!")
    elif max_of_all < 1e-3:
        print(f"   ✓ All samples match (small numerical differences)")
    else:
        print(f"   ✗ SIGNIFICANT DIFFERENCES DETECTED!")

    print("\n7. Feature names comparison...")
    expected_feature_names = list(test_processed.drop(['Label', 'Attack'], axis=1).columns)
    preprocessor_feature_names = preprocessor.get_feature_names()

    print(f"   Expected features: {len(expected_feature_names)}")
    print(f"   Preprocessor features: {len(preprocessor_feature_names)}")

    if expected_feature_names == preprocessor_feature_names:
        print(f"   ✓ Feature names match perfectly!")
    else:
        missing_in_preprocessor = set(expected_feature_names) - set(preprocessor_feature_names)
        extra_in_preprocessor = set(preprocessor_feature_names) - set(expected_feature_names)

        if missing_in_preprocessor:
            print(f"   ✗ Missing in preprocessor: {missing_in_preprocessor}")
        if extra_in_preprocessor:
            print(f"   ✗ Extra in preprocessor: {extra_in_preprocessor}")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_preprocessing_consistency()
