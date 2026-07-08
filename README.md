# EIT_Intubation_ML

MATLAB code for processing simulated 3D EIT voltage and conductivity data for

CSU Mueller EIT Lab: <https://www.engr.colostate.edu/laboratories/eit/>.

Copyright (c) 2026 Jonah Spector


# Simple-EIT

> **Electrical impedance tomography software for custom four-electrode device**


[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python
3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)

Electrical Impedance Tomography (EIT) is a method of medical imaging which
visualizes the internal resistivity of the human body, most commonly for
real-time monitoring of the lungs. It measures voltage on n-1 pairs of
electrodes, and sends an alternating current through the remaining pair of
electrodes. The pair of electrodes which produces the alternating current is
then replaced with a different pair of electrodes until all possible
combinations of electrodes are exhausted. This results in a non-linear,
ill-posed boundary value problem which, when solved, gives a 2D tomogram of
resistivities. We created a 4 electrode EIT device which utilizes machine
learning algorithms running locally to predict the location of an object of
higher resistivity (OHR) within one of 16 sectors. This probability distribution
is then visualized on a computer monitor in real time.

![Demo Day](./doc/final_report/title_image.jpg)

Research was performed using the device on multiple objects of different radii:

![CURC Poster](./doc/Simple_EIT_CURC_Poster.png)

*This project is no longer under active development.*

---

## Overview

Simple-EIT is a complete software stack and machine learning pipeline that:

- **Acquires voltage data** from 6 electrode configurations using multiplexed
  hardware
- **Classifies measurements** using trained ML models to identify OHR location
- **Visualizes results** in a real-time circular 16-sector display
- **Supports multiple classifiers**: SVM (recommended), Random Forest, XGBoost,
  Neural Networks, and more
- **Caches trained models** for each object to avoid redundant training
- **Auto-calibrates new objects** by collecting small datasets for ML training
  real-time

```
┌──────────────────────────────────────────────────┐
│  Simple-EIT Architecture                         │
├──────────────────────────────────────────────────┤
│  main.py           → GUI and orchestration       │
│  simple_eit.py     → EIT control layer           │
│  device_manager.py → PyVISA instrument control   │
│  classifier.py     → ML model management         │
│  data_collector.py → Training data collection    │
└──────────────────────────────────────────────────┘ 
```

---

## Hardware Requirements

| Component            | Model/Specification                      | Purpose                              |
| -------------------- | ---------------------------------------- | ------------------------------------ |
| Digital Oscilloscope | Keysight (any VISA-compatible)           | RMS voltage measurement              |
| Waveform Generator   | Agilent/Keysight (any VISA-compatible)   | Excitation signal                    |
| Analog Multiplexers  | 2× MUX36D04EVM-PDK                       | Electrode selection (S+, S-, V+, V-) |
| DC Power Supply      | 10 V DC                                  | Power multiplexer evaluation boards  |
| Controller           | Raspberry Pi 3B (recommended)            | GPIO control and computation         |
| 3D Printed Test Rig  | PETG                                     | Hold OHR and water                   |
| Electrodes           | 4x Stainless steel electrodes            | Measure voltage/generate current     |
| Connectors           | (16 + 2 + 2 + 4)x Banana hook connectors | Wire devices together                |

> **Safety Note**: Always verify Raspberry Pi can not recieve 10 V from DC power
> supply.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ihsotas0/simple-eit.git
cd simple-eit
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Raspberry Pi Setup
```bash
# Setup for PyVISA to allow USB connection
sudo groupadd usbgroup
sudo usermod -aG usbgroup $USER
sudo bash -c 'cat > /etc/udev/rules.d/99-usbgroup.rules <<EOF
   SUBSYSTEM=="usb", GROUP="usbgroup", MODE="0666"
   EOF'
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo reboot

# Connected Keysight/Agilent wavegen and scope

# Identify VID and PID of USB device
lsusb # Find VID and PID of wavegen and scope here e.g. idVendor1 and idVendor2, etc.
sudo bash -c 'cat > /etc/udev/rules.d/99-wavegen.rules <<EOF
   SUBSYSTEM=="usb", ATTR{idVendor1}=="VID", ATTR{idProduct1}=="PID", GROUP="usbgroup", MODE="0666"
   EOF'
sudo bash -c 'cat > /etc/udev/rules.d/99-scope.rules <<EOF
   SUBSYSTEM=="usb", ATTR{idVendor2}=="VID", ATTR{idProduct2}=="PID", GROUP="usbgroup", MODE="0666"
   EOF'
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo reboot
```

---

## Quick Start

### Run the Main Application
```bash
cd src
python main.py
```

1. **Enter instrument identifiers** when prompted (e.g., `DSOX`, `EDU`)
2. **Use keyboard controls** to interact:

| Key     | Action                                          |
| ------- | ----------------------------------------------- |
| `a`-`e` | Select predefined object (`curc_a` to `curc_e`) |
| `f`     | Run auto-calibration for new object             |
| `g`     | Set custom object name                          |
| `1`-`8` | Switch ML model (see table below)               |
| `x`     | Exit application                                |

### Available ML Models
| Key | Model                        | Description                                                |
| --- | ---------------------------- | ---------------------------------------------------------- |
| `1` | Gradient Boosting            | `sklearn.ensemble.GradientBoostingClassifier`              |
| `2` | K-Nearest Neighbors          | `sklearn.neighbors.KNeighborsClassifier`                   |
| `3` | Linear Discriminant Analysis | `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` |
| `4` | Logistic Regression          | `sklearn.linear_model.LogisticRegression`                  |
| `5` | Multi-Layer Perceptron       | `sklearn.neural_network.MLPClassifier`                     |
| `6` | Random Forest                | `sklearn.ensemble.RandomForestClassifier`                  |
| `7` | **Support Vector Machine**   | `sklearn.svm.SVC` (recommended)                            |
| `8` | XGBoost                      | `xgboost.XGBClassifier`                                    |

### Collect Training Data

> This can also be performed while the device runs by pressing `f`.

```bash
cd src
python data_collector.py
```

Choose:
- **`i`** → Collect instrument characterization data across frequencies
- **`o`** → Collect object-specific data (prompts for object name and location
  positioning)

> **Data Collection Workflow**:
> 1. Place test object at prompted location (AB_1, AB_2, ..., BC_4)
> 2. Press Enter to confirm position
> 3. System collects `n` measurements (default: 200)
> 4. Repeat for all 16 locations
> 5. CSV saved to `../data/{object_name}_data.csv`

---

## Project Structure

```
simple-eit/
├── LICENSE                # MIT License
├── README.md              # This file
├── todo.txt               # Future features
├── requirements.txt       # Python dependencies
│
├── src/                   # Source code
│   ├── main.py            # Entry point: GUI, threading, visualization
│   ├── simple_eit.py      # High-level EIT control wrapper
│   ├── classifier.py      # ML model management & inference
│   ├── device_manager.py  # PyVISA hardware interface
│   ├── pyvisa...tool.py   # PyVISA diagnostics tool for debugging
│   ├── data_collector.py  # Training data collection utilities
│   └── visualization.py   # CURC figure code
│
├── data/                  # Datasets & cached models
│   ├── *_data.csv         # Training datasets (generated)
│   ├── models/            # Cached .joblib model files
│   └── archive/           # Old data and models used for CURC
│
├── cad/                   # Test rig and OHR design files
│   └── *.step, *.stl
│
└── doc/                   # Extended documentation
```

---

## Configuration

### Python Constants

Most Python files associated with this project are configured with Python
constants like: `DEFAULT_WAVEGEN_IDN` which can be changed before running
`main.py`. Other ML models can be added to `model_factory` in `classifier.py`.

### Model Caching
Models are automatically cached using a hash of the training dataset:

```
data/models/{object_name}_{model_name}_{dataset_hash}.joblib
```

To force retraining, delete the corresponding `.joblib` file. Hashing of the
datasets ensures models are retrained while running `main.py` if new data is
appended to dataset while running auto-calibration.

---

## License

Distributed under the **MIT License**.

```
Copyright (c) 2026 Jonah Spector, Connor Cassidy, Chris Rayner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgments

Chuck Duey, Dr. Elaine Linde, Dr. Jennifer Mueller, Dr. Diego Krapf, Prof.
Olivera Notaros, Alaa Jallad, Nicholas Green, and Jennifer Kreinbrink.

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/ihsotas0/simple-eit/issues)
- **Authors**: 
  - Jonah Spector ([@ihsotas0](https://github.com/ihsotas0))
  - Connor Cassidy
  - Chris Rayner
