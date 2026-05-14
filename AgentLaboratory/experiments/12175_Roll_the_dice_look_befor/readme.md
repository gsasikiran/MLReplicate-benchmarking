# Explorations in Algorithmic Creativity via Next-Token and Multi-Token Approaches

This repository accompanies our research report on balancing coherence, creative diversity, and minimized memorization in language generation models. Our study systematically compares traditional next-token prediction (NTP) with multi-token teacherless prediction (MTP) and discrete diffusion methods (SEDD) across minimal yet representative combinatorial tasks.

---

## Table of Contents

- [Overview](#overview)
- [Research Motivation](#research-motivation)
- [Methods and Techniques](#methods-and-techniques)
  - [Seed-Conditioning](#seed-conditioning)
  - [Temperature Scaling](#temperature-scaling)
  - [Alignment Loss](#alignment-loss)
- [Experimental Setup](#experimental-setup)
- [Results and Discussion](#results-and-discussion)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Citations and Related Work](#citations-and-related-work)
- [License](#license)

---

## Overview

Our work addresses key challenges in algorithmic creativity for text generation. Traditional next-token prediction (NTP) methods, while coherent under deterministic conditions, are prone to memorization and low diversity. In contrast, our hybrid framework integrates:

- **Multi-Token Teacherless Prediction (MTP)**
- **Discrete Diffusion with Seed-Conditioning (SEDD)**

accompanied by controlled randomization methods (such as temperature scaling) to significantly lower memorization (from 100% to near 0%) while boosting output diversity (up to a metric of 1.00).

---

## Research Motivation

Algorithmic creativity in NLP must balance:
- **Coherence:** How well the output fits the desired semantic constraints.
- **Memorization:** Avoiding verbatim reproduction of training examples.
- **Diversity:** Generating novel and varied outputs.

We tackle these trade-offs on minimal combinatorial tasks such as:
- Sibling Discovery
- Triangle Discovery
- Circle Construction
- Line Construction

Through rigorous experimentation and ablation studies on synthetic datasets, our framework demonstrates that both MTP and SEDD outperform conventional NTP methods in achieving high creative output.

---

## Methods and Techniques

### Seed-Conditioning

- Injects controlled randomness at the input layer.
- Encourages exploration of latent representations.
- Reduces exact memorization even at low sampling temperatures.

### Temperature Scaling

- Utilizes a sampling temperature parameter (T) to adjust stochasticity.
- Adds noise based on the relation:  
  **p<sub>noise</sub> = min(0.9, α × T)**
  - α values by method:  
    • NTP: 0.3  
    • MTP: 0.5  
    • SEDD: 0.7

### Alignment Loss

- Enforces semantic consistency between a restrictive prompt (P<sub>r</sub>) and an adaptive prompt (P<sub>a</sub>).
- Defined as:  
  **L<sub>mix</sub> = 1 − cos(E(P<sub>r</sub>), E(P<sub>a</sub>))**
- Integrated into the overall loss:  
  **L<sub>total</sub> = L<sub>gen</sub> + λ L<sub>mix</sub>**

---

## Experimental Setup

- **Dataset:** Synthetic dataset with 200 examples designed for minimal combinatorial tasks.
- **Tasks:** Each task (e.g., Sibling Discovery, Triangle Discovery) follows task-specific validity criteria.
- **Generation Configuration:**
  - Experiments performed using three models: NTP, MTP, and SEDD.
  - Evaluated over 20 examples with 5 output generations per example.
  - Ablations include varying seed prefix lengths, temperature values (T=0, 0.5, 1.0, 2.0) and use of seed-conditioning.
  
A sample evaluation table under various conditions:

| Method + Condition            | Coherence (%) | Memorization (%) | Diversity |
|-------------------------------|---------------|------------------|-----------|
| NTP, T=0, Seed                | 54            | 49               | 69        |
| NTP, T=0, No Seed             | 80            | 100              | 20        |
| MTP, T=1.0, No Seed           | 69            | 1                | 100       |
| SEDD, T=1.0, No Seed           | 66            | 0                | 99        |

The standard transformer architecture is used across all settings for fair comparison, and the diffusion component in SEDD employs iterative denoising steps based on gradient updates.

---

## Results and Discussion

Key findings include:

- **Trade-Offs:**  
  • NTP exhibits high coherence but suffers from high memorization and low diversity.  
  • MTP and SEDD showcase near-zero memorization and robust diversity with a slight sacrifice in coherence.

- **Seed-Conditioning Impact:**  
  Prepending a random seed to the input disrupts verbatim replication, leading to higher creative output.

- **Temperature Effects:**  
  Increasing T results in improved diversity with lower memorization across the board following the p<sub>noise</sub> formulation.

- **Iterative Refinement:**  
  The diffusion process in SEDD helps in stepwise latent planning to ensure structured and creative outputs.

Our discussion emphasizes the necessity of balancing deterministic precision with controlled randomness. Although slight coherence degradation is observed when introducing high noise levels, our framework’s overall performance on synthetic combinatorial tasks is promising. Future work aims to extend these methods to more complex, real-world datasets (e.g., summarization tasks from XSUM, CNN/DailyMail) to further validate scalability and robustness.

---

## Repository Structure

A suggested directory layout:

```
├── data/                  # Synthetic datasets and examples.
├── experiments/           # Scripts to run NTP, MTP, and SEDD experiments.
├── notebooks/             # Jupyter notebooks for result visualization and ablation studies.
├── models/                # Model definitions and diffusion modules.
├── reports/               # LaTeX sources for the research paper.
├── README.md              # This file.
└── requirements.txt       # Dependency list.
```

---

## Getting Started

1. **Clone the Repository**

   ```
   git clone https://github.com/your-username/algorithmic-creativity.git
   cd algorithmic-creativity
   ```

2. **Install Dependencies**

   Ensure you have Python 3.8+ installed. Then run:

   ```
   pip install -r requirements.txt
   ```

3. **Run Experiments**

   Execute the experiments for a given method (e.g., MTP):

   ```
   python experiments/run_mtp.py --temperature 1.0 --use_seed True
   ```

4. **View Results**

   Results (e.g., coherence, memorization, diversity metrics) and logs are stored under the `experiments/` folder. Use the provided notebooks in `notebooks/` to visualize the ablation studies and trends.

---

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Other dependencies as listed in [requirements.txt](requirements.txt)

---

## Citations and Related Work

Our framework builds on previous studies highlighting the limitations of next-token prediction for creative generation. Key references include:
- arXiv:2504.15266v4 – Analysis of memorization and creativity in next-token prediction.
- arXiv:2403.06996v1 – Multi-token approaches for enhanced generation.
- arXiv:2401.10934v1 & arXiv:2308.12059v2 – Approaches integrating inpainting and prompt manipulation for creative tasks.

Please refer to the `reports/` directory for the full research paper and an extended literature review.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

We welcome contributions and discussions to further advance the field of algorithmic creativity. Thank you for your interest!

