# ARC Prize 2025 - Neural Architecture with Hailo-8 NPU

![AutomataNexus](https://img.shields.io/badge/AutomataNexus-AI-06b6d4?labelColor=64748b)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-c51a4a)
![ARC Prize](https://img.shields.io/badge/ARC%20Prize-2025%20Competition-FF6B6B)
![Prize Pool](https://img.shields.io/badge/Prize%20Pool-$1M-4ECDC4)
![AI Models](https://img.shields.io/badge/AI%20Models-5%20Custom%20Neural%20Networks-45B7D1)
![NPU](https://img.shields.io/badge/Hailo--8%20NPU-26%20TOPS-2ECC71)
![Pattern Detectors](https://img.shields.io/badge/Pattern%20Detectors-8%20Specialized-FFD93D)
![Status](https://img.shields.io/badge/Status-Active%20Development-success)

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24-013243?logo=numpy&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Training-F9AB00?logo=googlecolab&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Export-005CED?logo=onnx&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🏆 Overview

This repository contains our solution for the **ARC Prize 2025** - a $1,000,000 competition to solve Abstract Reasoning Corpus (ARC-AGI) tasks. Our approach combines **5 custom neural networks** with **8 specialized pattern detectors**, leveraging the **Hailo-8 NPU** (26 TOPS) for high-performance inference.

## 🎯 Competition Goal

Achieve **85% accuracy** on the ARC-AGI-2 private evaluation dataset to win the grand prize of **$700,000**.

## 🚀 Key Innovation

Unlike traditional approaches limited by Kaggle's 12-hour runtime, we use:
- **Offline Pre-computation**: Unlimited time on Hailo-8 NPU to analyze patterns
- **8 Specialized Pattern Detectors**: Matching our Apollo Nexus architecture
- **Fast Online Inference**: Pre-computed patterns enable rapid solving on Kaggle

## 📁 Project Structure

```
ARCPrize2025/
├── models/                    # Neural network models
│   ├── arc_models.py         # 5 custom neural networks
│   ├── train_arc_models_colab.ipynb  # Training notebook
│   └── docs/                 # Model documentation & COAs
├── arc_solver.py             # Main solver for Kaggle evaluation
├── pattern_detectors.py      # 8 specialized pattern detection modules
├── precompute_patterns.py    # Offline pattern analysis script
├── kaggle_submission.py      # Kaggle notebook submission script
├── colab_training.py         # Google Colab training script
├── explore_arc_data.py       # Data exploration and visualization
├── test_framework.py         # Local testing and validation
├── data/                     # ARC dataset files
│   ├── arc-agi_training_challenges.json
│   ├── arc-agi_evaluation_challenges.json
│   └── ...
├── docs/                     # Documentation
│   ├── architecture.md       # System architecture with diagrams
│   ├── pattern_detectors.md  # Pattern detector documentation
│   ├── testing_plan.md       # Testing strategy
│   └── ...
└── visualizations/           # Task visualizations
```

## 🤖 Neural Network Models

Our solution features 5 custom PyTorch models, each specialized for different reasoning tasks:

### 1. **MINERVA** - Strategic Pattern Analysis
- Vision Transformer architecture with pattern memory bank
- 8.7M parameters
- Handles strategic reasoning and decision making

### 2. **ATLAS** - Spatial Transformations  
- Spatial Transformer Network (STN)
- 3.5M parameters
- Specializes in geometric transformations and structural analysis

### 3. **IRIS** - Color Pattern Recognition
- Attention-based color relationship analyzer
- 4.2M parameters
- Masters color mappings and harmony detection

### 4. **CHRONOS** - Temporal Sequences
- Bidirectional LSTM with evolution prediction
- 6.1M parameters
- Tracks pattern evolution and sequences

### 5. **PROMETHEUS** - Creative Pattern Generation
- Variational Autoencoder (VAE) architecture
- 9.3M parameters
- Generates novel pattern solutions

## 🧠 Pattern Detectors

Supporting our neural networks are 8 specialized pattern detectors:

1. **GeometricDetector** - Rotations, reflections, translations
2. **ColorDetector** - Color mappings and transformations
3. **CountingDetector** - Numerical and size-based patterns
4. **LogicalDetector** - Boolean operations and conditionals
5. **SpatialDetector** - Gravity, alignment, boundaries
6. **SymmetryDetector** - Symmetry creation and completion
7. **ObjectDetector** - Object extraction and manipulation
8. **CompositeDetector** - Multi-step transformations

## 🔧 Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy
- Hailo SDK (for pre-computation)
- 16GB RAM minimum
- DELPHI device with Hailo-8 NPU (for offline phase)

## 📊 Performance

- **Pattern Detection**: <100ms per task
- **Pre-computation**: 2-4 hours for 1000 training tasks
- **Kaggle Runtime**: ~2 hours for 240 test tasks
- **Target Accuracy**: 85%+

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/AutomataControls/Arc2025.git
cd Arc2025

# Install dependencies
pip install -r requirements.txt

# Download ARC dataset
# Place in /data directory
```

## 🚦 Usage

### 1. Offline Pattern Pre-computation (on DELPHI/Hailo-8)

```bash
python precompute_patterns.py
```

This generates `precomputed_patterns.pkl` containing all discovered patterns.

### 2. Local Testing

```bash
python test_framework.py
```

Validates accuracy on evaluation data before submission.

### 3. Kaggle Submission

1. Upload `precomputed_patterns.pkl` as Kaggle dataset
2. Run `kaggle_submission.py` notebook
3. Submit generated `submission.json`

## 📈 Current Progress

- [x] Data exploration and analysis
- [x] 8 pattern detectors implemented
- [x] Pre-computation pipeline ready
- [x] Kaggle submission framework complete
- [ ] Full pattern library generation
- [ ] 85% accuracy threshold achieved

## 👥 Team

- **Andrew Jewell Sr.** - Lead Developer
- **AutomataNexus, LLC** - Organization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Competition

- **ARC Prize 2025**: $1,000,000 total prize pool
- **Grand Prize**: $700,000 for 85% accuracy
- **Deadline**: Check [official competition page](https://arcprize.org/)

## 🔗 Links

- [ARC Prize Official Site](https://arcprize.org/)
- [Kaggle Competition](https://www.kaggle.com/competitions/arc-prize-2025)
- [Hailo AI](https://hailo.ai/)

---

<div align="center">

**Built with ❤️ by AutomataNexus, LLC**

</div>