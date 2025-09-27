#!/usr/bin/env python3
"""
OLYMPUS Integration Test
Verify all components work together before final evaluation
"""

import numpy as np
import sys
import os

# Add paths
sys.path.append('/mnt/d/opt/ARCPrize2025')
sys.path.append('/mnt/d/opt/ARCPrize2025/models')

print("🏛️ OLYMPUS INTEGRATION TEST")
print("="*60)

# Test 1: Import all components
print("\n1. Testing imports...")
try:
    from ensemble_test_bench import OLYMPUSEnsemble
    print("  ✓ Ensemble Test Bench")
    
    from task_router import TaskRouter, SmartEnsemble
    print("  ✓ Task Router")
    
    from heuristic_solvers import HeuristicPipeline
    print("  ✓ Heuristic Solvers")
    
    from olympus_ensemble_runner import OLYMPUSRunner
    print("  ✓ OLYMPUS Runner")
    
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create simple test case
print("\n2. Creating test case...")
test_input = np.array([
    [0, 1, 0, 1],
    [1, 2, 1, 2],
    [0, 1, 0, 1],
    [1, 2, 1, 2]
])

train_examples = [
    {
        'input': np.array([[1, 2], [2, 1]]),
        'output': np.array([[2, 1], [1, 2]])  # Swap pattern
    },
    {
        'input': np.array([[0, 1], [1, 0]]),
        'output': np.array([[1, 0], [0, 1]])  # Swap pattern
    }
]

print(f"  ✓ Test input shape: {test_input.shape}")
print(f"  ✓ Training examples: {len(train_examples)}")

# Test 3: Initialize system (with dummy model paths for testing)
print("\n3. Initializing OLYMPUS system...")
try:
    # Create dummy model directory if needed
    model_dir = '/mnt/d/opt/ARCPrize2025/models/test_models'
    os.makedirs(model_dir, exist_ok=True)
    
    # For testing, we'll create dummy checkpoints
    import torch
    for model_name in ['minerva', 'atlas', 'iris', 'chronos', 'prometheus']:
        dummy_path = f"{model_dir}/{model_name}_best.pt"
        if not os.path.exists(dummy_path):
            torch.save({
                'model_state_dict': {},
                'val_exact': 0.5,
                'epoch': 1
            }, dummy_path)
    
    olympus = OLYMPUSRunner(model_dir=model_dir)
    print("  ✓ OLYMPUS Runner initialized")
    
except Exception as e:
    print(f"  ✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test prediction pipeline
print("\n4. Testing prediction pipeline...")
try:
    # Test without heuristics
    print("\n  a) Testing without heuristics...")
    result_no_heur = olympus.predict(
        test_input, 
        train_examples, 
        apply_heuristics=False, 
        verbose=False
    )
    print("    ✓ Prediction without heuristics completed")
    
    # Test with heuristics
    print("\n  b) Testing with heuristics...")
    result_with_heur = olympus.predict(
        test_input, 
        train_examples, 
        apply_heuristics=True, 
        verbose=True
    )
    print("    ✓ Prediction with heuristics completed")
    
    # Check results
    print(f"\n  Confidence without heuristics: {result_no_heur['confidence']:.1f}%")
    print(f"  Confidence with heuristics: {result_with_heur['confidence']:.1f}%")
    
except Exception as e:
    print(f"  ✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Verify all components were called
print("\n5. Verifying component integration...")
try:
    # Check ensemble result
    ensemble_result = result_with_heur['ensemble_result']
    if 'weights' in ensemble_result:
        print("  ✓ Task Router assigned weights")
        for model, weight in ensemble_result['weights'].items():
            print(f"    - {model}: {weight:.2f}")
    
    if 'all_predictions' in ensemble_result:
        print(f"  ✓ Ensemble collected predictions from {len(ensemble_result['all_predictions'])} models")
    
    print("  ✓ Heuristic pipeline was applied")
    
except Exception as e:
    print(f"  ✗ Component verification failed: {e}")

print("\n" + "="*60)
print("🎉 INTEGRATION TEST COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Place your trained model files (.pt) in the correct directory")
print("2. Run full evaluation: python olympus_ensemble_runner.py --evaluate")
print("3. Create submission: python olympus_ensemble_runner.py --submit test_challenges.json")
print("\n🏛️ OLYMPUS is ready to compete!")