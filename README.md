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



