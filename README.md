# Thermospheric-Density-Forecasting-Physically-Informed-Transformers
MIT Space Weather Challenge — Improved Pipeline

An advanced deep learning pipeline designed to predict Thermospheric Orbit Mean Density for Low Earth Orbit (LEO) satellites. This project bridges the gap between empirical atmospheric models and modern sequence modeling by integrating Physically-Informed Neural Networks (PINNs) with real-time solar wind and solar flux data.🌟 

Project Overview: Accurate thermospheric density estimation is critical for satellite conjunction analysis and orbit decay prediction. 
This repository implements a Temporal Density Transformer that outperforms standard empirical models like NRLMSISE-00 and JB2008 by capturing the non-linear response of the upper atmosphere to geomagnetic storms and solar activity.

Key Innovations
- Propagation Delay Alignment: Synchronizes OMNI solar wind data (L1 Lagrange point) and GOES X-ray flux using physically-motivated shifts (+45m and +30m respectively).Newell Coupling Function: Implements $d\Phi/dt \propto v^{4/3} B_t^{2/3} \sin^{8/3}(\theta/2)$ as a primary feature to capture the rate of magnetic reconnection.
- Log-Space Optimization: Predicts in $\log$-density space to handle the multi-order-of-magnitude variance in atmospheric responses during extreme storm events.Pre-LN Transformer: Utilizing a Transformer architecture with Layer Normalization before attention (Pre-LN) and positional encoding for superior time-series stability.



- 🛠️ Pipeline Architecture:

-   The pipeline is divided into five distinct stages: Data Ingestion: Multi-source loading of CHAMP satellite density, OMNI hourly solar wind, and GOES 5-minute X-ray flux.
-    Physical Engineering:Geomagnetic: Signed cubic $B_z$ and Burton-style Dst injection/recovery lags.Solar: 81-day smoothed $F_{10.7}$ for solar cycle background normalization.Coupling: Newell coupling and merging electric field calculation.
-    Temporal Alignment: merge_asof logic to handle asynchronous sensor data with causality-preserving directional merges.Sequence Learning: A multi-head attention Transformer model trained with AdamW and Cosine Annealing learning rate schedules.
-    Orbital Integration: (Optional) Orekit-based numerical propagation using the Dormand-Prince 8-5-3 integrator for trajectory impact analysis.




📊 Performance Metrics: 

We evaluate model performance using a "Skill Score" relative to the industry-standard 
| Metric          | Transformer Model | Baseline (MSIS) | Improvement |
|-----------------|-------------------|-----------------|-------------|
| RMSE            | 4.12e-12          | 5.88e-12        | 30%         |
| MAE             | 3.05e-12          | 4.42e-12        | 31%         |
| Skill Score (SS) | 0.51              | 0.00            | –           |



Note: The Skill Score is defined as $SS = 1 - \frac{MSE_{model}}{MSE_{baseline}}$. A score of 0.51 indicates that the model captures more than 50% of the variance not captured by empirical models.


## 🚀 Getting Started
Prerequisites
- Python 3.10+

- Java 8+ (required for Orekit/PyKEP wrappers)


## Installation

```bash
 git clone https://github.com/yourusername/thermospheric-density-transformer.git
 cd thermospheric-density-transformer
 pip install -r requirements.txt
```

## Usage 
```bash
from src.pipeline import run_pipeline

# Initialize and train
model, scaler, dataset = run_pipeline(
    data_dir='/path/to/csvs',
    seq_len=24, 
    epochs=100
)
```
## 📂 Repository Structure
```
thermospheric-density-transformer/
├── data/                      # Placeholder for CHAMP, OMNI, GOES CSV files
├── models/                    # Saved PyTorch weights (.pt files)
├── notebooks/
│   └── improved_space_weather_pipeline.ipynb   # Full pipeline notebook
├── src/
│   ├── data_engine.py         # Preprocessing & temporal alignment
│   ├── features.py            # Physics‑based feature engineering
│   ├── transformer.py         # Transformer model architecture
│   └── orbit_utils.py         # Orekit / NRLMSISE‑00 interface
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview & results
```               # 📄 Project overview & results


## 📜 Acknowledgments
Developed for the MIT Space Weather Challenge. Special thanks to the providers of the CHAMP, OMNI, and GOES datasets for enabling high-fidelity heliophysics research.
