#!/usr/bin/env python
"""Test script for the new nan_bypass functionality in batch_reshape."""

import sys
import os

import jax.numpy as jnp
import numpy as np
from feedbax.misc import batch_reshape


def simple_transform(x):
    """A simple transformation function for testing."""
    # Just multiply by 2 and add 1
    return 2 * x + 1


def test_nan_bypass():
    """Test the nan_bypass functionality."""
    print("Testing nan_bypass functionality...")

    # Create test data with batch dimensions and some NaN rows
    batch_shape = (3, 4)
    feature_dim = 5

    # Create data: batch shape (3, 4) with feature dimension 5
    data = np.random.randn(3, 4, feature_dim)

    # Insert some NaN rows
    data[0, 1, :] = np.nan  # Row at batch position (0, 1) is all NaN
    data[1, 2, 2:] = np.nan  # Row at batch position (1, 2) has partial NaN
    data[2, 0, :] = np.nan  # Row at batch position (2, 0) is all NaN

    print(f"Input shape: {data.shape}")
    print(f"NaN pattern in flattened view (first 12 rows):")
    data_flat = data.reshape(-1, feature_dim)
    for i in range(min(12, len(data_flat))):
        has_nan = np.any(np.isnan(data_flat[i]))
        print(f"  Row {i}: {'has NaN' if has_nan else 'valid'}")

    # Test without nan_bypass (should work since simple_transform handles NaN)
    print("\n--- Testing without nan_bypass ---")
    transform_normal = batch_reshape(simple_transform)
    result_normal = transform_normal(data)
    print(f"Result shape: {result_normal.shape}")

    # Test with nan_bypass
    print("\n--- Testing with nan_bypass=True ---")
    transform_nan_bypass = batch_reshape(simple_transform, nan_bypass=True)
    result_nan_bypass = transform_nan_bypass(data)
    print(f"Result shape: {result_nan_bypass.shape}")

    # Check that NaN rows are preserved in the same positions
    result_flat = result_nan_bypass.reshape(-1, feature_dim)
    print("NaN pattern in result (first 12 rows):")
    for i in range(min(12, len(result_flat))):
        has_nan = np.any(np.isnan(result_flat[i]))
        print(f"  Row {i}: {'has NaN' if has_nan else 'valid'}")

    # Verify that valid rows are correctly transformed
    print("\n--- Verifying correct transformation of valid rows ---")
    data_flat = data.reshape(-1, feature_dim)
    result_flat = result_nan_bypass.reshape(-1, feature_dim)

    for i in range(len(data_flat)):
        input_has_nan = np.any(np.isnan(data_flat[i]))
        result_has_nan = np.any(np.isnan(result_flat[i]))

        if input_has_nan:
            if not result_has_nan:
                print(f"ERROR: Row {i} had NaN input but non-NaN result!")
        else:
            if result_has_nan:
                print(f"ERROR: Row {i} had valid input but NaN result!")
            else:
                # Check if transformation is correct
                expected = 2 * data_flat[i] + 1
                if not np.allclose(result_flat[i], expected, rtol=1e-6):
                    print(f"ERROR: Row {i} transformation incorrect!")
                    print(f"  Input: {data_flat[i]}")
                    print(f"  Expected: {expected}")
                    print(f"  Got: {result_flat[i]}")

    print("Test completed!")


def test_all_nan_input():
    """Test case where all rows contain NaN."""
    print("\n=== Testing all-NaN input ===")

    # Create data where all rows contain NaN
    data = np.full((2, 3, 4), np.nan)

    transform_nan_bypass = batch_reshape(simple_transform, nan_bypass=True)
    result = transform_nan_bypass(data)

    print(f"Input shape: {data.shape}")
    print(f"Result shape: {result.shape}")
    print(f"All result values are NaN: {np.all(np.isnan(result))}")


def test_no_nan_input():
    """Test case where no rows contain NaN."""
    print("\n=== Testing no-NaN input ===")

    # Create data with no NaN values
    data = np.random.randn(2, 3, 4)

    transform_nan_bypass = batch_reshape(simple_transform, nan_bypass=True)
    result = transform_nan_bypass(data)

    expected = 2 * data + 1

    print(f"Input shape: {data.shape}")
    print(f"Result shape: {result.shape}")
    print(f"Transformation correct: {np.allclose(result, expected, rtol=1e-6)}")



def pca_like_transform(x):
    """A transform that fails on NaN like sklearn PCA would."""
    # Check for NaN and raise error (like sklearn would)
    if jnp.any(jnp.isnan(x)):
        raise ValueError("Input contains NaN values")

    # Simple mock PCA transformation: project to first 2 components
    # Just use the first 2 features as a simple projection
    return x[..., :2]


def test_pca_like_with_nan_bypass():
    """Test that nan_bypass allows PCA-like functions to work with NaN data."""
    print("Testing PCA-like transform with NaN data...")

    # Create test data with NaN rows
    data = np.random.randn(2, 3, 5)
    data[0, 1, :] = np.nan  # One full row is NaN
    data[1, 0, 2:] = np.nan  # Another row has partial NaN

    print(f"Input shape: {data.shape}")

    # Test without nan_bypass - should fail
    print("\n--- Testing without nan_bypass (should fail) ---")
    transform_normal = batch_reshape(pca_like_transform)
    try:
        result = transform_normal(data)
        print("ERROR: Expected failure but succeeded!")
    except ValueError as e:
        print(f"Expected failure: {e}")

    # Test with nan_bypass - should work
    print("\n--- Testing with nan_bypass=True (should work) ---")
    transform_with_bypass = batch_reshape(pca_like_transform, nan_bypass=True)

    try:
        result = transform_with_bypass(data)
        print(f"Success! Result shape: {result.shape}")

        # Check that NaN pattern is preserved
        result_flat = result.reshape(-1, result.shape[-1])
        data_flat = data.reshape(-1, data.shape[-1])

        print("Checking NaN pattern preservation:")
        for i in range(len(data_flat)):
            input_has_nan = np.any(np.isnan(data_flat[i]))
            result_has_nan = np.any(np.isnan(result_flat[i]))

            if input_has_nan != result_has_nan:
                print(f"ERROR: Row {i} NaN pattern not preserved!")
            else:
                status = "NaN preserved" if input_has_nan else "valid transformed"
                print(f"  Row {i}: {status}")

    except Exception as e:
        print(f"Unexpected failure: {e}")


if __name__ == "__main__":
    test_pca_like_with_nan_bypass()
    test_nan_bypass()
    test_all_nan_input()
    test_no_nan_input()

