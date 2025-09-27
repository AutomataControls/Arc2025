# Claude Assistant Prompt for ARC Prize 2025

## Context
You are helping with the ARC Prize 2025 competition - a $1,000,000 challenge to solve Abstract Reasoning Corpus (ARC-AGI) tasks. This project uses a Hailo-8 NPU (26 TOPS) on a DELPHI device for offline pattern pre-computation to achieve 85% accuracy needed for the $700,000 grand prize.

## Project Overview
- **Competition**: ARC Prize 2025 - Abstract reasoning puzzle solving
- **Goal**: 85% accuracy on private evaluation set
- **Hardware**: Hailo-8 NPU on DELPHI device (Raspberry Pi + Hailo)
- **Strategy**: Pre-compute patterns offline, fast inference online
- **Architecture**: 8 pattern detectors matching Apollo Nexus design

## Key Files
1. **pattern_detectors.py** - 8 specialized pattern detection modules:
   - GeometricDetector (rotations, reflections, translations)
   - ColorDetector (color mappings)
   - CountingDetector (numerical patterns)
   - LogicalDetector (boolean operations)
   - SpatialDetector (gravity, alignment)
   - SymmetryDetector (symmetry patterns)
   - ObjectDetector (object manipulation)
   - CompositeDetector (multi-step transformations)

2. **precompute_patterns.py** - Runs on Hailo-8 to analyze all training tasks
3. **arc_solver.py** - Main solver using pre-computed patterns
4. **kaggle_submission.py** - Submission script for Kaggle
5. **test_framework.py** - Local testing and validation

## Current Task Priorities
1. **Pattern Library Generation**:
   - Run precompute_patterns.py on all 1000 training tasks
   - Utilize Hailo-8 NPU parallel processing
   - Generate precomputed_patterns.pkl (~50MB)

2. **Testing and Validation**:
   - Test accuracy on evaluation subset
   - Identify weak pattern types
   - Improve detection algorithms

3. **Optimization**:
   - Ensure <12 hour Kaggle runtime
   - Achieve 85%+ accuracy
   - Handle edge cases

## Technical Details
- **Input**: JSON tasks with 'train' and 'test' examples
- **Output**: JSON predictions with 'attempt_1' and 'attempt_2'
- **Grid Size**: Typically 3x3 to 30x30
- **Colors**: 0-9 (10 possible values)
- **Patterns**: Geometric, color, counting, logical, spatial, symmetry, object-based

## Commands
```bash
# Test pattern detectors
python3 pattern_detectors.py

# Run pre-computation (2-4 hours)
python3 precompute_patterns.py

# Test locally
python3 test_framework.py

# Prepare Kaggle submission
python3 kaggle_submission.py
```

## Hardware Utilization
- Use Hailo-8's 26 TOPS for parallel pattern analysis
- Run 8 detectors simultaneously during pre-computation
- Cache all discovered patterns for fast lookup

## Success Criteria
- Pattern library covers >80% of task types
- Pre-computation completes in <4 hours
- Kaggle submission runs in <2 hours  
- Accuracy reaches 85% threshold

## Important Notes
- We have multiple submission attempts (not just one)
- Each submission provides 2 attempts per task
- Kaggle has no internet access during evaluation
- Pre-computed patterns are our key advantage

When assisting, focus on:
1. Maximizing Hailo-8 NPU utilization
2. Improving pattern detection accuracy
3. Optimizing for Kaggle constraints
4. Achieving 85% accuracy target