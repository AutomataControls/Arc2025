# ARC Prize 2025 - Hailo-Powered Pattern Recognition

<div align="center">

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Hailo](https://img.shields.io/badge/Hailo-8_NPU-00D4AA?style=for-the-badge)](https://hailo.ai/)
[![Status](https://img.shields.io/badge/Status-In_Development-orange?style=for-the-badge)](https://github.com/AutomataControls/Arc2025)

</div>

## 🏆 Overview

This repository contains our solution for the **ARC Prize 2025** - a $1,000,000 competition to solve Abstract Reasoning Corpus (ARC-AGI) tasks. Our approach leverages the **Hailo-8 NPU** (26 TOPS) for offline pattern pre-computation, enabling fast and accurate solving during Kaggle evaluation.

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
├── arc_solver.py              # Main solver for Kaggle evaluation
├── pattern_detectors.py       # 8 specialized pattern detection modules
├── precompute_patterns.py     # Offline pattern analysis script
├── kaggle_submission.py       # Kaggle notebook submission script
├── explore_arc_data.py        # Data exploration and visualization
├── test_framework.py          # Local testing and validation
├── data/                      # ARC dataset files
├── docs/                      # Documentation
│   ├── architecture.md        # System architecture with diagrams
│   ├── pattern_detectors.md   # Pattern detector documentation
│   ├── testing_plan.md        # Testing strategy
│   └── ...
└── visualizations/            # Task visualizations
```

## 🧠 Pattern Detectors

Our solution uses 8 specialized detectors inspired by the Apollo Nexus architecture:

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