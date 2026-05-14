# Conformal Prediction as Bayesian Quadrature for Risk Control

This repository contains the code, experiments, and supporting materials for the research report “Conformal Prediction as Bayesian Quadrature for Risk Control.” The project introduces a novel framework that leverages Bayesian quadrature to reformulate conformal prediction, providing rigorous, data-conditional, and distribution‐free risk guarantees for high-stakes prediction tasks.

## Overview

In high-stakes applications such as medical diagnostics, autonomous driving, and financial forecasting, it is imperative to have robust uncertainty quantification with explicit risk control guarantees. Traditional conformal prediction methods, while distribution-free, may yield overly optimistic or conservative decisions. Our approach reformulates conformal prediction through the lens of Bayesian quadrature by introducing an aggregated loss

  L⁺ = ∑₍ᵢ₌₁₎⁽ⁿ⁺¹⁾ Uᵢ ℓ₍(ᵢ)₎

where the weights Uᵢ are sampled from a Dirichlet distribution (U ∼ Dir(1, …, 1)) with ℓ₍ₙ₊₁₎ set to a worst-case bound B. This formulation allows us to verify the condition

  Pr(L⁺ ≤ α) ≥ β

ensuring that predictive risk is rigorously controlled. The framework not only recovers standard methods like Split Conformal Prediction (SCP) and Conformal Risk Control (CRC) as special cases, but also introduces a novel high posterior density (HPD) rule that exploits the full posterior via Monte Carlo Dirichlet sampling.

## Repository Contents

- **/code**: Implementation of the Bayesian quadrature-based conformal prediction methods.
  - **hpd_rule.py**: Contains the implementation of the High Posterior Density (HPD) rule.
  - **methods.py**: Implements various conformal prediction methods including SCP, CRC, HPD, and RCPS.
  - **utils.py**: Utility functions for Monte Carlo Dirichlet sampling and other supporting computations.
- **/experiments**: Experimental setups and scripts for reproducing results.
  - **binomial_experiment.ipynb**: Jupyter notebook for the synthetic binomial loss experiment.
  - **regression_experiment.ipynb**: Jupyter notebook for the heteroskedastic regression task.
  - **multilabel_experiment.ipynb**: Notebook and scripts for multilabel classification (e.g., MS-COCO simulation).
- **/docs**: Supporting documents including the research report (in LaTeX) and supplementary notes.
  - **research_report.tex**: The full LaTeX source of the research paper.
- **README.md**: This file, outlining the repository and instructions for use.

## Features

- **Robust Risk Control**: Provides a mechanism to control predictive risk by integrating over the quantile function of the loss distribution.
- **Method Comparisons**: Includes implementations of standard methods (SCP, CRC) alongside the proposed HPD rule and a uniform concentration based rule (RCPS).
- **Extensible Framework**: The modular design allows adaptation to various loss functions and applications, including regression and multilabel classification tasks.
- **Reproducible Experiments**: Detailed scripts and notebooks enable easy replication of our synthetic experiments and ablation studies on calibration sample size, candidate grid resolution, and confidence levels.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.10 or later
- NumPy
- SciPy
- PyTorch (or TensorFlow if preferred for certain experiments)
- Jupyter Notebook (for running the provided notebooks)
- (Optional) Additional libraries as specified in the `requirements.txt` (if provided)

### Installation

1. Clone or download the repository:
   ```
   git clone https://github.com/yourusername/conformal-bq-risk-control.git
   cd conformal-bq-risk-control
   ```

2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

*Note: If a `requirements.txt` file is not provided, please install the dependencies manually using pip.*

## Running Experiments

The experiments are set up as Jupyter notebooks. You can run them using:
```
jupyter notebook
```
Then, open the desired notebook from the **/experiments** folder:
- **binomial_experiment.ipynb**: Reproduces the synthetic binomial loss experiment.
- **regression_experiment.ipynb**: Runs the heteroskedastic regression task.
- **multilabel_experiment.ipynb**: Demonstrates the multilabel classification setting using synthetic data mimicking MS-COCO scores.

Each notebook details:
- The setup and parameter configuration (e.g., candidate thresholds, risk levels such as α and confidence levels β).
- The execution of the conformal prediction methods (SCP, CRC, HPD, and RCPS).
- Visualization and summary tables of the results (average selected thresholds, test risks, and failure rates).

## Project Structure and Code Overview

- **hpd_rule.py**: Implements the HPD rule with Monte Carlo sampling over Dirichlet weights, helping to compute credible bounds on the aggregated loss L⁺.
- **methods.py**: Provides functions for computing order statistics for SCP, posterior mean based estimators for CRC, and the uniform concentration bound (RCPS).
- **utils.py**: Houses helper functions including sorting of losses, Dirichlet sampling, and risk evaluation metrics.

The code is designed to be modular so new loss functions or risk evaluation criteria can be added with minimal modifications.

## Experimental Results

Our experiments demonstrate the following key findings:

- **Synthetic Binomial Loss Experiment**:
  - SCP often produces candidate thresholds with an average value around 0.596 and a high failure rate (~61.6%).
  - The CRC method improves risk control with an average candidate threshold of around 0.771 (failure rate ~1.9%).
  - The proposed HPD rule robustly selects a candidate threshold (≈ 0.970) with a 0% failure rate, achieving a balanced trade-off between risk control and utility.
  - The RCPS method, while conservative (threshold = 1.000), also results in 0% failure but at the cost of prediction set utility.

- **Heteroskedastic Regression Experiment**:
  - A similar trend is observed, with HPD achieving superior risk control and lower test risk compared to SCP and CRC.
  - Detailed performance metrics, including test risk and failure rates, are provided in the experiment notebooks.

## Future Work

Planned extensions of this work include:
- **Non-i.i.d. Settings**: Extending the framework to handle calibration losses with dependencies, such as in time-series or spatial data.
- **Heavy-Tailed Loss Distributions**: Integrating robust approaches to handle heavy-tailed loss scenarios.
- **Model-Aware Training**: Embedding the risk control mechanism within the model training process for end-to-end risk-aware prediction.
- **Scalability Enhancements**: Optimizing the Monte Carlo sampling process for large-scale datasets and distributed computing environments.

## Citation

If you use this code in your research, please consider citing our work:
> Agent Laboratory. “Conformal Prediction as Bayesian Quadrature for Risk Control.” (Year). [Link to paper if available].

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please contact [your-email@example.com] or open an issue in the repository.

---

Happy experimenting and thank you for your interest in our work!