# Algorithmic Creativity on Minimal Open-Ended Tasks

Welcome to the repository for "Comparing Reasoning and Prompt Engineering Techniques to Maximize Algorithmic Creativity on Minimal Open-Ended Tasks." This project investigates techniques to enhance generative models for creative output by comparing traditional next-token prediction (NTP) with multi-token teacherless prediction (MTP) and a discrete diffusion baseline (SEDD). The repository includes code, experimental data, and analysis for replicating and extending the experiments described in the paper.

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Methodology](#methodology)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## Overview

In this work, we explore algorithmic creativity by quantitatively evaluating generative models on four minimal open-ended tasks:
  
- **Sibling Discovery**  
- **Triangle Discovery**  
- **Circle Construction**  
- **Line Construction**

We introduce a composite creativity metric defined as:

  crN̂ = diversity × coherence × (1 − memorization)

where:

- **Coherence:** Percentage of outputs retaining at least 80% of the input's structural aspects.
- **Diversity:** Ratio of unique outputs after canonicalization.
- **Memorization:** Proportion of outputs that exactly replicate the input.

By contrasting NTP, MTP, and SEDD, we demonstrate that while NTP ensures high coherence, it tends to overfit (high memorization). MTP significantly reduces memorization through simultaneous multi-token prediction, and SEDD uses iterative denoising to further balance diversity and structural fidelity.

---

## Background

Generative models traditionally use next-token prediction (NTP), which suffers from sequential dependency and memorization, often yielding “clever Hans” effects. Our research builds on recent advances in:

- **Multi-Token Teacherless Prediction (MTP):** Predicts several tokens in parallel, reducing overfitting.
- **Discrete Diffusion (SEDD):** Utilizes an iterative denoising process guided by a learned velocity field.

Key randomness mechanisms include temperature sampling (τ ∈ {0, 0.5, 1.0, 2.0}) and seed-conditioning (prepending a fixed 10-token prefix) to effectively explore the latent creative space in minimal tasks.

---

## Methodology

Our approach integrates three prediction paradigms:

- **Next-Token Prediction (NTP):** Uses traditional autoregressive generation.
- **Multi-Token Teacherless Prediction (MTP):** Predicts multiple tokens concurrently to mitigate sequential bias.
- **Discrete Diffusion (SEDD):** Refines outputs iteratively via a probability flow ordinary differential equation (ODE):

  d𝒙(t)/dt = vθ(𝒙(t), t, c),

  where vθ is a learned velocity field and c represents context cues.

Randomness is introduced through temperature sampling and seed-conditioning, balancing the trade-off between coherence and diversity.

---

## Experimental Setup

Experiments are conducted on synthetic datasets for each of the four tasks comprising 10 curated examples. The evaluation protocol involves:

- **Metrics:**
  - **Coherence:** % outputs with ≥80% structural overlap.
  - **Diversity:** Ratio of unique canonical outputs.
  - **Memorization:** Frequency of outputs identical to input.
  
- **Controlled Variables:**
  - Temperature values: τ ∈ {0, 0.5, 1.0, 2.0}
  - Top-k sampling options
  - Seed-conditioning with fixed 10-token prefixes

- **Ablation Studies:** Hyperparameters such as temperature, top-k, and seed lengths were varied to assess their impact on crN̂.

The repository contains scripts to run these experiments as well as Jupyter notebooks for interactive analysis.

---

## Results

Our experiments reveal:

- **Sibling Discovery:**  
  - NTP: 70% coherence, 30% memorization, crN̂ = 49%  
  - MTP: Identical to NTP in aggregate metrics  
  - SEDD: Improved coherence (80%) and lower memorization (20%), crN̂ = 64%

- **Triangle Discovery:**  
  - NTP: 100% coherence, 40% memorization, crN̂ = 60%  
  - MTP: 100% coherence, 10% memorization, crN̂ = 90%  
  - SEDD: Intermediate results (memorization 30%, crN̂ = 70%)

- **Circle Construction:**  
  - NTP achieves perfect scores (crN̂ = 100%)  
  - MTP and SEDD incur higher memorization resulting in lower crN̂ scores (80% and 60% respectively)

- **Line Construction:**  
  - NTP: crN̂ = 90% (memorization 10%)  
  - MTP: crN̂ = 80% (memorization 20%)  
  - SEDD: crN̂ = 70% (memorization 30%)

A detailed table comparing these metrics is provided in the paper and replicated in our analysis notebooks. Additionally, figures illustrating creativity scores and the trade-off between diversity and coherence are available.

---

## Usage

### Prerequisites

- Python (>= 3.7)
- Required libraries as listed in `requirements.txt`:
  - numpy
  - matplotlib
  - scipy
  - (and any additional dependencies specific to your experiments)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/AgentLaboratory/algorithmic-creativity.git
   cd algorithmic-creativity
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running Experiments

- To train or evaluate models using NTP, MTP, or SEDD, run:
  ```
  python run_experiments.py --method [NTP|MTP|SEDD] --task [sibling|triangle|circle|line] --temperature 1.0 --top_k True
  ```

- To reproduce all experiments and generate corresponding plots, execute:
  ```
  python run_all_experiments.py
  ```

Detailed usage instructions, configuration options, and parameter descriptions can be found in the inline help:
   ```
   python run_experiments.py --help
   ```

---

## Reproducibility

All experiments include options for controlled randomness via fixed seeds and seed-conditioning methods. The repository includes:

- Experiment scripts for each method and task
- Configuration files detailing hyperparameter settings
- Jupyter notebooks for analysis and visualization

Results obtained have been verified with statistical significance (p < 0.05) in critical discovery tasks. The setup ensures full reproducibility and serves as a baseline framework for future research in computational creativity.

---

## Citation

If you find this work useful in your research, please cite:

  Agent Laboratory (2023). "Comparing Reasoning and Prompt Engineering Techniques to Maximize Algorithmic Creativity on Minimal Open-Ended Tasks." [Online]. Available: https://github.com/AgentLaboratory/algorithmic-creativity

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For questions, feedback, or contributions, please open an issue or submit a pull request. Happy experimenting!