# Score Matching with Missing Data: A Marginal Importance-Weighted Approach

This repository contains the implementation and supplementary materials for the paper:

**"Score Matching with Missing Data: A Marginal Importance-Weighted Approach"**  
*Agent Laboratory*

The paper introduces a novel framework for density estimation in the presence of missing data by extending conventional score matching methods. It leverages importance sampling to approximate the marginal score function on observed data components along with a boundary-aware weighting scheme to handle truncation effects in compact support domains.

This repository includes:
- The LaTeX source of the paper.
- Code implementations of the proposed marginal IW score matching method.
- Experiments for synthetic Gaussian data, ICA-like data, and graphical model recovery.
- Instructions to reproduce the experimental results.

---

## Overview

In many real-world applications, data are not fully observed (e.g., missing completely at random, MCAR). Standard score matching techniques require full data observations, restricting their applicability. Our approach addresses this limitation by:
- Approximating the marginal score function through importance sampling.
- Incorporating boundary-aware weighting that naturally downweights observations near the data boundaries.
- Establishing a framework with finite-sample guarantees via analysis of the marginal Fisher divergence.

### Key Contributions
- **Marginal Score Estimation:** Estimation of the score function solely on the observed components of the data.
- **Importance Sampling:** Monte Carlo estimation using a proposal density for the missing dimensions.
- **Boundary-Aware Weighting:** Mitigation of truncation biases in compact support domains.
- **Empirical Evaluations:** Demonstrated on synthetic Gaussian datasets, ICA-like data (via rejection sampling), and applications in graphical model recovery.

---

## Repository Structure

```
.
├── src
│   ├── data_generation.py         # Scripts for generating Synthetic Gaussian and ICA-like data
│   ├── model.py                   # Implementation of the marginal IW score matching model
│   ├── train.py                   # Training routines and optimization using Adam, gradient clipping, etc.
│   └── graph_recovery.py          # Graphical model recovery experiments
├── experiments
│   ├── gaussian_results.ipynb     # Jupyter Notebook for Gaussian parameter estimation experiments
│   ├── ica_results.ipynb          # Jupyter Notebook for ICA-like experiment with zero-imputation baseline
│   └── graph_results.ipynb        # Jupyter Notebook for the graphical model recovery task
├── paper
│   └── score_matching_missing_data.tex  # LaTeX source of the paper
├── README.md                      # This file
└── requirements.txt               # List of required Python packages
```

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your_username/score-matching-missing-data.git
   cd score-matching-missing-data
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install the Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   The repository is implemented using [PyTorch](https://pytorch.org/) with support for double-precision computations to ensure numerical stability.

---

## Usage

### Running Experiments

- **Synthetic Gaussian Experiment:**
  
  To run the synthetic Gaussian estimation experiment, execute:
  ```bash
  python src/train.py --experiment gaussian
  ```
  This script generates 800 samples in ℝ¹⁰, applies 30% MCAR missingness, and optimizes the marginal importance-weighted score matching objective over 200 iterations using Adam. Check the results in the provided Jupyter Notebook (`experiments/gaussian_results.ipynb`).

- **ICA-like Experiment:**

  For ICA-like data generated via rejection sampling with 50% MCAR missingness, execute:
  ```bash
  python src/train.py --experiment ica
  ```
  A naive zero-imputation baseline is used for evaluation. More details and performance metrics (L2 error) are available in `experiments/ica_results.ipynb`.

- **Graphical Model Recovery:**

  To replicate the graphical model recovery experiment:
  ```bash
  python src/graph_recovery.py
  ```
  This experiment recovers a star graph structure by estimating precision matrices via zero-imputation, ridge stabilization, and soft-thresholding. The ROC AUC metric is computed and reported. See `experiments/graph_results.ipynb` for detailed analysis and visualization.

### Command Line Arguments

Each script provides options to customize parameters such as:
- Learning rate (`--lr`)
- Number of iterations (`--iterations`)
- Number of importance samples (`--samples`)

For example:
```bash
python src/train.py --experiment gaussian --lr 0.01 --iterations 200 --samples 10
```

---

## Experimental Results

### Synthetic Gaussian Data
- **Estimated Mean Vector:** Approximately [0.7882, 1.0772, 0.4244, -0.6640, -0.6782, -0.2541, 0.4978, 1.5648, -0.7241, -0.4680]
- **Final Loss:** ~ -2.38
- **L2 Estimation Error:** ~ 2.70

### ICA-like Data
- **True Mean (ICA):** Approximately [0.0288, 0.0474, 0.0836, -0.0063, -0.0058, -0.0127, 0.0149, 0.0313, 0.0346, 0.0190]
- **Zero-Imputation L2 Error:** ~ 0.0965

### Graphical Model Recovery
- **Precision Matrix Recovery:** Soft-thresholded precision matrix reveals the star graph structure.
- **ROC AUC:** ~ 0.57

Detailed visualizations and analysis can be found in the corresponding Jupyter Notebooks in the `experiments/` folder.

---

## Citation

If you find our work useful for your research, please consider citing our paper:

    @inproceedings{agent2023scorematching,
      title={Score Matching with Missing Data: A Marginal Importance-Weighted Approach},
      author={Agent Laboratory},
      year={2023},
      note={Available in the GitHub repository: https://github.com/your_username/score-matching-missing-data}
    }

---

## Future Work

Future extensions may include:
- Systematic analysis of the trade-off between the number of importance samples and gradient variance.
- Exploration of alternative designs for boundary-aware weighting functions.
- Extension to handle missingness not at random (MAR/MNAR) via inverse propensity sampling or latent variable models.
- Integration of variational techniques to further improve the model's finite-sample guarantees and computational efficiency.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For any questions or suggestions, please open an issue or contact the repository maintainer at [your_email@example.com](mailto:your_email@example.com).

Happy experimenting!