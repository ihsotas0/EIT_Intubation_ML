# EIT_Intubation_ML

> **MATLAB code for processing simulated 3D EIT voltage and conductivity data
> into infant intubation status classification using deep learning.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![MATLAB 2026a](https://img.shields.io/badge/MATLAB-2026a-blue)](https://www.mathworks.com/products/matlab.html)

---

## Overview

---

## Hardware Requirements

---

## Installation

---

## Quick Start

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

---

## License

Distributed under the **MIT License**.

```
Copyright (c) 2026 Jonah Spector

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

## Contact

- **Issues**: [GitHub Issues](https://github.com/ihsotas0/simple-eit/issues)
- **CSU Mueller EIT Lab**: [Link](https://www.engr.colostate.edu/laboratories/eit/)
- **Authors**: 
  - Jonah Spector ([@ihsotas0](https://github.com/ihsotas0))
  - Kyler Howard
