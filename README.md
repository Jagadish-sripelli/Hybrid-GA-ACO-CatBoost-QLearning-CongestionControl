# Hybrid Bio-Inspired and Machine Learning Framework for Real-Time Congestion Control

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange)]()

> **A Resource-Efficient Hybrid Bio-Inspired and Machine Learning Framework for Real-Time Congestion Control in Heterogeneous Networks**

---

## Overview

This repository contains the full implementation of the hybrid congestion control framework proposed in the paper. The framework integrates:

- **Genetic Algorithm (GA)** — global path exploration
- **Ant Colony Optimization (ACO)** — pheromone-guided local path refinement
- **CatBoost** — proactive congestion prediction (Normal / Moderate / Critical)
- **Q-Learning** — decentralized adaptive forwarding control
- **Hybrid Decision Engine** — composite routing score combining all four components

---

## Repository Structure

```
hybrid-congestion-control/
│
├── src/
│   ├── ga_aco/
│   │   ├── genetic_algorithm.py       # GA route optimization
│   │   └── ant_colony.py              # ACO pheromone routing
│   ├── catboost_predictor/
│   │   ├── train_catboost.py          # CatBoost model training
│   │   └── predict_congestion.py      # Real-time congestion prediction
│   ├── q_learning/
│   │   └── q_agent.py                 # Distributed Q-learning agents
│   ├── hybrid_engine/
│   │   └── decision_engine.py         # Hybrid routing decision engine
│   └── utils/
│       ├── network_graph.py           # Network topology utilities
│       ├── traffic_generator.py       # Synthetic traffic generator
│       └── metrics.py                 # QoS metrics computation
│
├── experiments/
│   ├── run_grid_topology.py           # Grid topology experiment
│   ├── run_random_topology.py         # Random topology experiment
│   ├── run_scalefree_topology.py      # Scale-free topology experiment
│   └── run_ablation_study.py          # Ablation study
│
├── datasets/
│   ├── synthetic/                     # Generated synthetic traffic
│   ├── caida_sample/                  # CAIDA trace sample (see note)
│   └── mawi_sample/                   # MAWI trace sample (see note)
│
├── results/
│   ├── figures/                       # All paper figures (PNG + PDF)
│   └── tables/                        # All paper tables (CSV)
│
├── configs/
│   └── params.yaml                    # All algorithm parameters
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_results_visualization.ipynb
│   └── 04_statistical_validation.ipynb
│
├── tests/
│   └── test_all.py
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/hybrid-congestion-control.git
cd hybrid-congestion-control
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run a full experiment
```bash
# Grid topology, 100 nodes
python experiments/run_grid_topology.py --nodes 100 --runs 20

# Random topology, 150 nodes
python experiments/run_random_topology.py --nodes 150 --runs 20

# Scale-free topology, 100 nodes
python experiments/run_scalefree_topology.py --nodes 100 --runs 20
```

### 4. Run ablation study
```bash
python experiments/run_ablation_study.py
```

---

## Algorithm Parameters

All parameters are in `configs/params.yaml`. Key values from the paper:

| Parameter | Value | Component |
|---|---|---|
| Crossover probability (p_c) | 0.7 | GA |
| Mutation probability (p_m) | 0.1 (dynamic up to 0.25) | GA |
| Population size | 50 | GA |
| Generations | 100 per routing interval | GA |
| Number of ants | 20 | ACO |
| Pheromone evaporation (ρ) | 0.3 | ACO |
| Number of trees | 500 | CatBoost |
| Learning rate (η) | 0.05 | CatBoost |
| Max tree depth | 8 | CatBoost |
| Q-learning rate (α) | 0.2 | Q-Learning |
| Discount factor (γ) | 0.8 | Q-Learning |
| Simulation runs | 20 (independent seeds) | All |

---

## Key Results

| Method | Throughput (Mbps) | Latency (ms) | PDR (%) | Pkt Loss (%) |
|---|---|---|---|---|
| OSPF | 8.1 | 91 | 85.6 | 10.0 |
| GA–ACO | 8.7 | 72 | 81.2 | 13.7 |
| CatBoost–Q-L | 9.3 | 68 | 84.2 | 6.5 |
| DQN Routing | 9.8 | 64 | 85.6 | 13.7 |
| **Proposed** | **11.4** | **56** | **92.3** | **4.8** |

All results statistically validated at **p < 0.001** (one-way ANOVA + Tukey HSD).

---

## Datasets

| Dataset | Source | Usage |
|---|---|---|
| CAIDA | [caida.org](https://www.caida.org/catalog/datasets/passive_dataset/) | Real-world traffic traces |
| MAWI | [mawi.wide.ad.jp](http://mawi.wide.ad.jp/mawi/) | Real-world traffic traces |
| Synthetic CBR | Generated (`utils/traffic_generator.py`) | Controlled experiments |
| Synthetic Poisson | Generated (`utils/traffic_generator.py`) | Bursty traffic experiments |

> **Note:** CAIDA and MAWI datasets require free registration. Download and place in `datasets/caida_sample/` and `datasets/mawi_sample/`. Sample synthetic datasets are included.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hybrid_congestion_2025,
  title   = {A Resource-Efficient Hybrid Bio-Inspired and Machine Learning
             Framework for Real-Time Congestion Control in Heterogeneous Networks},
  author  = {[Authors]},
  journal = {[Journal Name]},
  year    = {2025},
  note    = {Under Review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
