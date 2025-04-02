import os
import argparse
import numpy as np
import h5py
from neurocombat_sklearn.neurocombat_sklearn import CombatModel

def generate_cache(cache_dir):
    # Generate random input data
    data = np.random.rand(100, 10)  # Example data
    sites = np.random.randint(1, 4, size=100).reshape(-1, 1)  # Example site labels reshaped to 2D

    # Save input data to HDF5 cache
    input_data_path = os.path.join(cache_dir, "combat_model_input.h5")
    with h5py.File(input_data_path, 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('sites', data=sites)

    # Initialize and fit CombatModel
    combat = CombatModel()
    combat.fit(data, sites)
    transformed_data = combat.transform(data, sites)

    # Save transformed data to HDF5 cache
    transformed_data_path = os.path.join(cache_dir, "combat_model_transformed.h5")
    with h5py.File(transformed_data_path, 'w') as f:
        f.create_dataset('transformed_data', data=transformed_data)

def test_against_cache(cache_dir):
    # Load cached input and output data
    input_data_path = os.path.join(cache_dir, "combat_model_input.h5")
    transformed_data_path = os.path.join(cache_dir, "combat_model_transformed.h5")

    with h5py.File(input_data_path, 'r') as f:
        cached_input_data = {
            'data': f['data'][:],
            'sites': f['sites'][:]
        }

    with h5py.File(transformed_data_path, 'r') as f:
        cached_output_data = f['transformed_data'][:]

    # Extract input data
    data = cached_input_data['data']
    sites = cached_input_data['sites']

    # Fit and transform using CombatModel
    combat = CombatModel()
    combat.fit(data, sites)
    transformed_data = combat.transform(data, sites)

    # Compare the transformed data with cached output
    np.testing.assert_allclose(transformed_data, cached_output_data, rtol=1e-5, atol=1e-8)
    print("Test passed: Transformed data matches cached output.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or test CombatModel caches.")
    parser.add_argument("--cache-dir", required=True, help="Directory to store or load cache files.")
    parser.add_argument("--action", required=True, choices=["generate", "test"], help="Action to perform: generate a new cache or test against an existing cache.")
    args = parser.parse_args()

    if args.action == "generate":
        generate_cache(args.cache_dir)
        print("Cache generated successfully.")
    elif args.action == "test":
        test_against_cache(args.cache_dir)
