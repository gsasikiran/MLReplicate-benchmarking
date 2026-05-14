# Ensemble-Based Bayesian Aggregation for Multiturn Human–LLM Collaboration

Welcome to the repository for our research on "Ensemble-Based Bayesian Aggregation with Uncertainty-Guided Clarifications for Multiturn Human–LLM Collaboration". This repository hosts the implementation, simulation scripts, and supporting materials for our approach that integrates ensemble Monte Carlo reward predictors, Bayesian meta-calibration, and dynamic uncertainty-driven clarification modules. The work aims to enhance long-term dialogue quality and task performance in various domains such as document editing, code generation, mathematical problem solving, and ambiguity resolution.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Setup and Results](#experimental-setup-and-results)
- [Discussion and Future Work](#discussion-and-future-work)
- [Citation](#citation)
- [License](#license)
- [Authors](#authors)

---

## Overview

This project addresses key challenges in multiturn human–LLM collaboration. Traditional reward-based LLM training often relies on immediate next-turn signals. Our approach overcomes these limitations by estimating a conversation-level reward:

  R*(t|g) = R_ext(t, g) + R_int(t)

where:
- R_ext(t, g) quantifies task-specific performance (e.g., BLEU scores or unit test pass rates).
- R_int(t) penalizes inefficiencies (based on token usage) and integrates an LLM-driven interactivity score.

An ensemble of Monte Carlo-based reward predictors (varying window sizes and sample counts) is employed. Their outputs are aggregated using Bayesian linear regression to produce both an aggregated reward and an uncertainty metric. When the estimated uncertainty exceeds a predefined threshold (e.g., τ = 0.15), the system triggers an uncertainty-guided clarification module to refine dialogue outcomes.

---

## Key Features

- **Ensemble of Monte Carlo Reward Predictors:** Leverages diverse configurations (e.g., window sizes w ∈ {1,2,3} and sample counts S ∈ {3,5}) for robust reward estimation.
- **Bayesian Meta-Calibration:** Uses Bayesian linear regression to aggregate ensemble outputs and quantify uncertainty.
- **Uncertainty-Guided Clarifications:** Dynamically triggers clarification rounds when the uncertainty (σ_agg) exceeds the set threshold, applying a bonus adjustment (δ = 0.05) to refine responses.
- **Domain Flexibility:** Experimental results span multiple domains:
  - Document Editing (MediumDocEdit-Chat)
  - Code Generation (BigCodeBench-Chat)
  - Mathematical Problem Solving (MATH-Chat)
  - Ambiguity Resolution (Abg-CoQA)
- **Active Learning Integration:** Incorporates a two-phase training pipeline including pretraining with synthetic dialogues and active fine-tuning, enhanced via LoRA (Low-Rank Adaptation).

---

## Repository Structure

```
├── README.md                     # This file
├── LICENSE                       # Repository license information
├── docs/                         # Detailed documentation, extended discussions, and experimental notes
├── experiments/                  # Scripts and notebooks for running experiments
│   ├── doc_editing/              # MediumDocEdit-Chat domain experiments
│   ├── code_generation/          # BigCodeBench-Chat domain experiments
│   ├── math_problem_solving/     # MATH-Chat experiments
│   └── ambiguity_resolution/     # Abg-CoQA experiments
├── ensemble/                     # Source code for ensemble MC reward predictors and Bayesian calibration module
│   ├── predictors.py             # Implementation of Monte Carlo reward predictors
│   ├── calibrator.py             # Bayesian linear regression calibration module 
│   └── clarification.py          # Uncertainty-guided clarification module
├── active_learning/              # Active fine-tuning routines, LoRA integration, and active learning scripts
└── synthetic_datasets/           # Synthetic datasets for various domains
```

---

## Installation

Before running the code, please ensure that your environment meets the following requirements:

- Python 3.8 or newer
- Required Python packages (install via pip):

```bash
pip install -r requirements.txt
```

*Note: The `requirements.txt` file contains dependencies such as NumPy, SciPy, scikit-learn, and PyTorch, among others.*

---

## Usage

The main entry points are provided through scripts placed in the `experiments/` and `active_learning/` directories. For example:

- To run an experiment on MediumDocEdit-Chat:
  ```bash
  python experiments/doc_editing/run_experiment.py
  ```
- To execute the Bayesian calibration with the ensemble:
  ```bash
  python ensemble/calibrator.py
  ```
- For active fine-tuning with uncertainty-guided clarifications:
  ```bash
  python active_learning/active_tuning.py
  ```

Additional configuration options (e.g., window sizes, Monte Carlo sample counts, threshold values) can be adjusted via the respective configuration files.

---

## Experimental Setup and Results

### Experimental Domains & Metrics
- **MediumDocEdit-Chat:** Evaluated using BLEU scores and average token counts.
- **BigCodeBench-Chat:** Evaluated via simulated unit test pass rates.
- **MATH-Chat:** Evaluated using final answer accuracy.
- **Abg-CoQA:** Evaluated using Macro Accuracy and F1 scores.

### Sample Results

| Domain                         | Baseline Metric       | Active (with Clarifications) |
| ------------------------------ | --------------------- | ---------------------------- |
| MediumDocEdit-Chat (BLEU)      | 0.625                 | 0.637                        |
| BigCodeBench-Chat (Unit Test)  | 0.532                 | 0.489                        |
| MATH-Chat (Accuracy)           | 0.739                 | 0.799                        |
| Abg-CoQA (Macro Accuracy/F1)   | 0.800                 | 1.000                        |

These results illustrate that targeted clarification can reduce ambiguity and improve dialogue quality, although trade-offs (e.g., in the code generation domain) highlight the need for careful tuning.

---

## Discussion and Future Work

Our work demonstrates that an ensemble-based approach combined with Bayesian uncertainty estimation and proactive clarifications yields notable improvements in multiturn human–LLM collaboration. While results in document editing, mathematical problem solving, and ambiguity resolution are promising, further work is needed to optimize dynamic thresholds and reduce potential negative impacts in rapid-response scenarios (e.g., code generation).

### Future Directions:
- **Adaptive Penalty Tuning:** Dynamically adjust the intrinsic cost parameters based on real-time feedback.
- **Dynamic Thresholding:** Develop adaptive strategies for uncertainty thresholds based on domain-specific requirements.
- **Enhanced Sampling Efficiency:** Investigate advanced sampling methods (e.g., importance or stratified sampling) to reduce computational overhead.
- **Personalized and Context-Aware Clarifications:** Incorporate user profiles and richer dialogue context features.
- **Robustness to Adversarial Inputs:** Further explore defense mechanisms against ambiguous or adversarial prompts.

For a detailed discussion and extended theoretical background, please refer to the documentation in the `docs/` directory.

---

## Citation

If you find this work useful in your research, please consider citing our paper:

Agent Laboratory. (2023). Ensemble-Based Bayesian Aggregation with Uncertainty-Guided Clarifications for Multiturn Human-LLM Collaboration. [arXiv preprint].

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Authors

- Research conducted at Agent Laboratory
- For questions or contributions, please open an issue or submit a pull request.

---

We appreciate your interest in our work and welcome contributions, suggestions, and feedback. Enjoy exploring the code and advancing human–LLM collaboration research!