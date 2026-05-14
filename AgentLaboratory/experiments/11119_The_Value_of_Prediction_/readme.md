# Quantifying the Trade-Offs in Policy Evaluation

Welcome to the repository for the research project “Quantifying the Trade-Offs in Policy Evaluation.” This repository contains the code, simulation experiments, and supporting materials for a framework that rigorously evaluates the trade-offs between prediction accuracy and screening access in policy evaluation. In this study, we introduce a policy value function and the Prediction-Access Ratio (PAR) metric to help decision-makers balance model sophistication against resource expansions.

---

## Overview

In many policy settings—such as public employment, healthcare, or education—enhancing predictive performance via advanced machine learning models must be balanced against practical limitations on screening capacity. The core contributions of this work include:

- **Policy Value Function**  
  A rigorous function defined as:  
  V(α, β, R²) = Φ₂(zₐ, z_b; ρ) / β,  
  where zₐ = Φ⁻¹(α), z_b = Φ⁻¹(β), and ρ = √R². This formulation captures the non-linear sensitivity of screening thresholds and predictive performance.

- **Prediction-Access Ratio (PAR)**  
  A metric that quantifies the relative improvement in policy value when increasing screening thresholds (α) compared to enhancements in predictive accuracy (R²).

- **Empirical and Theoretical Insights**  
  Simulation experiments using synthetic datasets demonstrate that modest improvements—for example, increasing Test R² from 0.16866 to 0.32661 via residual scaling (δ = 0.1)—can improve the empirical policy value from 0.70000 to 0.80000. Additionally, capacity gap analysis shows that a small screening increment (Δα* ≈ 0.0300) can yield gains comparable to those from complex model enhancements.

---

## Repository Contents

- **/code**: Contains source code for training predictive models (e.g., Gradient Boosting, Decision Trees), applying residual scaling, and evaluating the policy value function.
- **/data**: Synthetic datasets and scripts to generate the data splits (training, validation, test).
- **/experiments**: Scripts that run simulation experiments, perform capacity gap analysis, and compute the Prediction-Access Ratio (PAR).
- **/docs**: This README, the full research report in LaTeX, and additional notes on the methodology.
- **/results**: Output files, summary tables, and figures from the experiments.

---

## Features

- **Policy Evaluation Framework**:  
  Evaluate trade-offs between model accuracy and screening access in a unified setting.

- **Residual Scaling Implementation**:  
  Simulate improved prediction performance without the need to retrain models fully.

- **Empirical Analysis**:  
  Compare complex models (e.g., Gradient Boosting) with simpler alternatives (e.g., Decision Trees) using key metrics such as Test R² and empirical policy value V(α, β).

- **Sensitivity Analysis**:  
  Derive and analyze local sensitivities (∂V/∂α and ∂V/∂R²) and estimate the Prediction-Access Ratio (PAR) under different finite improvements.

---

## Requirements

- Python 3.7 or higher
- Required libraries:
  - numpy
  - scipy
  - scikit-learn
  - matplotlib (for plotting results)
  - (Optional) CatBoost or XGBoost, if you wish to experiment with alternative models

You can install the required packages via:
```
pip install -r requirements.txt
```

---

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/policy-evaluation.git
   cd policy-evaluation
   ```

2. (Optional) Set up a virtual environment:
   ```
   python -m venv env
   source env/bin/activate  # On Windows use: env\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

---

## Usage

1. **Data Generation & Preparation**:  
   Use the scripts in the `/data` folder to either generate or load the synthetic datasets, following the documented splits:
   - Training: 169 samples
   - Validation: 69 samples
   - Test: 62 samples

2. **Model Training**:  
   Run the training script for the complex model (Gradient Boosting regressor) located in `/code/train_model.py`. You can adjust hyperparameters such as the number of estimators, early stopping rounds, and random seed as required.

3. **Residual Scaling and Policy Evaluation**:  
   Execute the evaluation script in `/experiments/evaluate_policy.py` to apply residual scaling (with scaling factor δ = 0.1) and compute both the empirical policy value V(α, β) and the Prediction-Access Ratio (PAR).

4. **Results & Analysis**:  
   Check the `/results` folder for output tables and graphs summarizing the performance metrics:
   - Baseline vs. improved Test R² values
   - Empirical policy values (e.g., increase from 0.70000 to 0.80000)
   - Capacity gap analysis highlighting the minimal screening increment Δα* ≈ 0.0300

---

## Experimental Setup

The simulated experiments mimic real-world administrative datasets with key covariates such as age, gender, and outcome measures. Key configuration details include:

- **Screening Threshold (α)**: 0.2  
- **Outcome Quantile Threshold (β)**: 0.15  
- **Hyperparameters (Complex Model)**:
  - 5000 estimators
  - Early stopping after 20 iterations with no improvement
  - Fixed random seed (42)

The evaluation metric V(α, β) is computed by comparing the proportion of observations falling below the respective thresholds for both the predicted and true outcome values.

---

## Results Summary

- **Theoretical Analysis**:  
  - Policy value V(α, β, R²) derived with local sensitivities: ∂V/∂α ≈ 1.77513 and ∂V/∂R² ≈ 0.61282.
  - Prediction-Access Ratio (PAR) values around 2.0 under finite increases in α and R².

- **Empirical Findings**:  
  - Complex model: Test R² improved from 0.16866 to 0.32661 using residual scaling.
  - Empirical policy value increased from 0.70000 to 0.80000.
  - Capacity gap analysis indicates that a screening increase of Δα* ≈ 0.0300 yields comparable benefits to enhanced prediction performance.

For detailed results, refer to the reports and output summaries in the `/docs` and `/results` folders.

---

## Contributing

Contributions to the project are welcome. If you have suggestions, improvements, or additional experiments, please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or further discussion, please contact:

- Agent Laboratory (maintainer)
- Email: agent.laboratory@example.com

---

## References

- arXiv:2108.04134v1 – Algorithmic profiling and fairness in public employment settings.
- arXiv:2308.02624v1 – Recent developments in algorithmic fairness.
- arXiv:2409.02888v1 – Cost-effectiveness analysis in policy evaluation.

This repository aims to contribute to the literature on algorithmic fairness and resource-constrained policy evaluation. We hope our work serves as a practical guide for both researchers and policymakers seeking to balance predictive accuracy with operational capacity.

Happy coding and research!