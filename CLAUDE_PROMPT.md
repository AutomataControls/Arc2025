# Claude Assistant Prompt for ARC Prize 2025

## Context
You are helping with the ARC Prize 2025 competition - a $1,000,000 challenge to solve Abstract Reasoning Corpus (ARC-AGI) tasks. We achieved a BREAKTHROUGH with 0.68% exact match using the V4 training approach and are now executing a strategic plan to reach 85% accuracy for the $700,000 grand prize.

## Critical Credentials
- **GitHub Token**: ghp_HJfznfQwortLdVBPMkGRXQKNcQNZWQ0QZFrh
- **Competition**: ARC Prize 2025 (not 2024!)
- **Kaggle Username**: Already configured in ~/.kaggle/

## Breakthrough Status
- **MINERVA**: 0.68% exact match achieved! (V3 fixed training)
- **ATLAS**: Currently training with V4 mega-scale curriculum
- **Key Fix**: Transformation penalty was negative (-0.3), rewarding copying. Fixed to positive (+1.5)
- **Current GPU**: A100 80GB (only using 4.1GB of 80GB!)

## V4 Training Configuration (PROVEN TO WORK)
```python
# V4 MEGA-SCALE + CURRICULUM
BATCH_SIZE = 512
LEARNING_RATE = 0.01
OPTIMIZER = SGD with Nesterov momentum
SCHEDULER = CosineAnnealingLR
EPOCHS = 300 (100 per curriculum stage)

# Critical Loss Components
TRANSFORMATION_PENALTY = 0.5  # MUST BE POSITIVE!
EXACT_MATCH_BONUS = 5.0      # Sparse but powerful reward
```

## Strategic Roadmap to 85% - FOUR PILLARS

### Pillar 1: Maximize Ensemble Coverage (HIGHEST PRIORITY)
**The "Team of Specialists" Strategy**
- Train ALL 5 models: MINERVA ✓, ATLAS (in progress), IRIS, CHRONOS, PROMETHEUS
- Each specialist covers different pattern types
- OLYMPUS ensemble score = sum of all specialists minus overlap

**Task Router** (to build):
- Different grid sizes → ATLAS gets 1.5x weight
- Color changes → IRIS gets 1.5x weight  
- Sequences → CHRONOS gets 1.3x weight
- Complex multi-object → MINERVA gets 1.4x weight
- Novel patterns → PROMETHEUS gets 1.2x weight

### Pillar 2: Post-Processing Heuristics
**Fix "Almost Correct" Predictions**

1. **Symmetry Solver**: Snap almost-symmetric grids to perfect symmetry
2. **Object Integrity Solver**: Fill single-pixel holes, remove noise
3. **Grid Size Solver**: Match output size rules from training examples
4. **Color Palette Solver**: Ensure output only uses input colors

### Pillar 3: Enhanced Data Usage
1. **Color Permutation Augmentation**: Swap all colors consistently (teaches relationships)
2. **Pseudo-Labeling**: Use confident ensemble predictions on test set as new training data

### Pillar 4: Model Refinements
- Systematic hyperparameter tuning (Optuna/Ray Tune)
- Architectural experiments (deeper networks, explicit counting modules)

## Execution Phases

**Phase 1** (NOW): Finish training all 5 specialists
**Phase 2** (Parallel): Build Task Router + Heuristic Solvers
**Phase 3**: Create Ensemble Test Bench, evaluate true performance
**Phase 4**: Iterate based on failure analysis, implement pseudo-labeling

## Project Structure
```
/mnt/d/opt/ARCPrize2025/
├── models/
│   └── arc_models_enhanced.py (5 specialists: MINERVA, ATLAS, IRIS, CHRONOS, PROMETHEUS)
├── colab_training_v4_megascale_curriculum.py (CURRENT BEST TRAINER)
├── olympus_ensemble_clean.html (documentation)
├── kaggle_integration.py (submission tools)
└── data/ (competition data already downloaded)
```

## Key Insights from Training
1. **Curriculum Learning is CRITICAL**: Start at stage 0 (easy), not stage 2
2. **Smart Copying**: Models learn when copying IS correct (identity tasks)
3. **Volatility is Normal**: ARC tasks are diverse, metrics will fluctuate
4. **First Exact Match is Hardest**: We got 0.68%, now scale up

## Critical Bugs We Fixed
1. **TRANSFORMATION_PENALTY**: Was -0.3 (rewarding copying!), fixed to +0.5
2. **Curriculum Stage**: Was starting at 2 (hardest), fixed to 0
3. **Dropout in Decoders**: Was killing exact matches, removed ALL
4. **Scheduler**: Fixed ReduceLROnPlateau to track validation accuracy

## Commands
```bash
# Current training script (V4 Mega-Scale + Curriculum)
cd /mnt/d/opt/ARCPrize2025
python colab_training_v4_megascale_curriculum.py

# Kaggle operations
kaggle competitions files arc-prize-2025
kaggle competitions download arc-prize-2025
kaggle competitions submit arc-prize-2025 -f submission.json -m "OLYMPUS Ensemble"
```

## Important Rules
1. **NO MORE NEW TRAINING SCRIPTS** - Only fix existing V4
2. **NO SUMMARIES** unless explicitly asked
3. **POSITIVE transformation penalty** always
4. **Start curriculum at stage 0**
5. **Remove dropout from decoders**

## Success Metrics
- Individual model exact match > 1%
- Ensemble exact match > 10% 
- Post-processing adds +5-10%
- Final target: 85% = $700,000

When assisting, focus on:
1. Training remaining specialists (IRIS, CHRONOS, PROMETHEUS)
2. Building Task Router for smart weighting
3. Implementing post-processing heuristics
4. Creating Ensemble Test Bench
5. Analyzing failures to guide improvements