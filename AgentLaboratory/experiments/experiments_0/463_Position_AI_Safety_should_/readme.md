# Iterative Prompt Engineering for AI Labor Market Impact Simulation

This repository contains the code, data, and documentation for our research project that explores the use of iterative prompt engineering combined with hierarchical reasoning modules to simulate and quantify the multifaceted impacts of artificial intelligence (AI) on labor market dynamics. The repository supports both traditional econometric analyses and dynamic simulation techniques to model wage dynamics, labor displacement, and income distribution shifts induced by AI.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Setup and Results](#experimental-setup-and-results)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Contact](#contact)

---

## Overview

Our work presents an iterative framework that refines simulation prompts using evolutionary strategies. This approach achieves a significant improvement in simulation fidelity (from 0.419 to 0.660, approximately 57.6% improvement), allowing us to capture complex interdependencies between AI-induced labor dynamics and traditional market variables. The project includes:

- **Iterative Prompt Engineering:** An optimization layer that refines prompts via gradient-based evolutionary strategies.
- **Hierarchical Reasoning Modules:** Mechanisms to explore complex counterfactual scenarios relevant to policy simulation.
- **Econometric Analysis:** Implementation of models like difference-in-differences (DiD) and panel regression to quantify treatment effects and explore wage dynamics.
- **Performance Metrics:** Evaluation using metrics such as task correctness, time-to-solve, and ROC-AUC for content detectability.

The simulation framework serves as a decision-support tool for policymakers and researchers aiming to understand AI's impact on wage distributions and labor market adjustments.

---

## Features

- **Dual-Pipeline Simulation:**
  - **Baseline Pipeline:** Uses static prompt configurations for standard simulations.
  - **Optimized Pipeline:** Employs iterative prompt refinement to improve simulation fidelity.
  
- **Econometric Modeling:**
  - Wage dynamics modeled with the equation:  
    wₜ = α + β · Treatmentₜ + γ · Xₜ + εₜ
  - Generalized production function incorporating traditional capital, AI-induced capital, and labor.
  
- **Advanced Evaluation Metrics:**
  - Fidelity score improvement from 0.419 to 0.660.
  - Treatment effect estimation (β ≈ 0.754 with p ≈ 0.056).
  - Negative impact of AI concentration on wage share (coefficient ≈ -12.388, p = 0.001).
  - Performance improvements in task correctness and reduction in time-to-solve.
  
- **Experimental Framework:**
  - Uses a license-clean, open-access subset of the HuggingFace `imdb` dataset for simulation purposes.
  - Detailed hyperparameter configuration (including learning rate η = 0.05 and simulation parameters).
  
- **Visualization:**
  - Time-series evolution of wage changes.
  - Boxplots for task correctness and processing time for baseline and optimized pipelines.

---

## Project Structure

The repository is organized as follows:

```
├── data/
│   ├── imdb_subset/           # License-clean, open-access data subset.
│   └── processed/             # Processed panel data for simulation.
│
├── docs/
│   ├── paper.pdf              # Draft of the research paper.
│   └── figures/               # Plots and charts (e.g., DiD results, performance boxplots).
│
├── src/
│   ├── simulation/            # Code for simulation pipelines.
│   │   ├── base_pipeline.py   # Baseline static prompt implementation.
│   │   ├── iterative_pipeline.py  # Iterative prompt optimization with evolutionary strategy.
│   │   └── hierarchical_reasoning.py  # Hierarchical reasoning modules.
│   │
│   ├── econometrics/          # Econometric models and analysis (DiD, panel regression, PCA).
│   │   ├── did_analysis.py
│   │   └── regression.py
│   │
│   └── utils/                 # Utility functions for data preprocessing, evaluation metrics, etc.
│
├── requirements.txt           # Python dependencies (e.g., statsmodels, scikit-learn, numpy, matplotlib).
├── README.md                  # This file.
└── LICENSE                    # Project license.
```

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/iterative-prompt-engineering.git
   cd iterative-prompt-engineering
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Data Preparation:**
   - Place the license-clean subset of the `imdb` dataset into the `data/imdb_subset/` directory.
   - Run the data pre-processing script from the `src/utils/` directory to convert raw text into panel data with treatment indicators and temporal events.

2. **Run the Baseline Simulation:**

   ```bash
   python src/simulation/base_pipeline.py
   ```

   This executes the static prompt simulation and outputs the baseline metrics (e.g., fidelity score, DiD treatment effect).

3. **Run the Optimized Simulation:**

   ```bash
   python src/simulation/iterative_pipeline.py
   ```

   This script runs the iterative prompt engineering optimization, updates the prompts, and evaluates the enhanced simulation fidelity.

4. **Econometric Analysis:**
   - For running the difference-in-differences analysis:
     ```bash
     python src/econometrics/did_analysis.py
     ```
   - For panel regression and other econometric evaluations:
     ```bash
     python src/econometrics/regression.py
     ```

5. **Visualization:**
   - Visualization scripts are available in the `docs/figures/` folder. They can be executed to generate updated charts (e.g., time-series evolution and boxplots for performance metrics).

---

## Experimental Setup and Results

The experimental setup follows these key steps:

- **Dataset:** Utilizes a curated subset of the HuggingFace `imdb` dataset mapped to generate panel data with treatment status.
- **Simulation:** Compares baseline static prompts vs. an optimized iterative prompt pipeline.
- **Metrics:** 
  - Prompt fidelity increases from 0.419 to 0.660.
  - DiD analysis shows a treatment effect coefficient of roughly 0.754 (p ≈ 0.056).
  - Regression on wage share exhibits a significant negative coefficient on AI concentration (≈ -12.388, p = 0.001).
  - Task performance improvements include enhanced correctness (0.657 to 0.783) and reduced time-to-solve (54.79s to 46.55s).
- **Economic Modeling:** Efficiency gains from iterative prompt updates are incorporated with classical econometric models to capture policy simulation scenarios.

For detailed experimental results, refer to the paper draft in [docs/paper.pdf](docs/paper.pdf) and the embedded tables and figures in the documentation.

---

## Contributing

Contributions are welcome! If you have suggestions or improvements, please follow these steps:

1. Fork the repository.
2. Create your feature branch: `git checkout -b feature/YourFeature`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/YourFeature`.
5. Open a pull request.

For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## References

- Research reports available on arXiv (e.g., arXiv:2401.09718v3, arXiv:2508.16603v1, arXiv:2412.07042v1, arXiv:2405.18369v2, arXiv:2412.18196v2).
- Related literature on prompt optimization frameworks: GreenTEA and PromptWizard.

---

## Contact

For questions or further discussion, please contact:

- **Project Lead:** Agent Laboratory
- **Email:** agentlab@university.edu

Feel free to open an issue in the repository for any queries or bug reports.

---

Happy simulating and exploring the intricate dynamics of AI and labor markets!