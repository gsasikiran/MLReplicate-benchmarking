# Adaptive Prompt-Enhanced Score Matching for Partially Observed Data

Welcome to the repository for the research project on **Adaptive Prompt-Enhanced Score Matching for Partially Observed Data**. This project presents a novel framework to recover score functions from datasets with significant missing data by dynamically adapting key hyperparameters using a meta-learning prompt generator. The repository contains code, experimental setups, and detailed reports that illustrate our methodology, experimental results, and discussion on future research directions.

---

## Overview

In many real-world applications—from biomedical data to large-scale image collections—data often exhibits partial observations due to sensor failures, privacy issues, or corruption. Traditional score matching techniques tend to struggle under these conditions, especially when naïve imputation is applied. Our work addresses these challenges by:

- **Adaptive Hyperparameter Tuning:** Leveraging a meta-learning prompt generator to dynamically select key hyperparameters (e.g., sample size, number of inner-loop variational steps, learning rates, and numerical stabilization parameters).
  
- **Robust Score Matching Strategies:** Implementing both marginal Importance-Weighted (Marg-IW) and marginal Variational (Marg-Var) approaches to effectively estimate the score function.

- **Numerical Stabilization:** Utilizing techniques such as log-sum-exp regularization and gradient clipping to ensure stable convergence during optimization under a Missing Completely at Random (MCAR) mechanism (with 30% missing entries).

- **Extensive Evaluations:** Conducting experiments on synthetic datasets including multivariate Gaussians, ICA-inspired models, and Gaussian Graphical Models (GGMs) with star graph structures.

---

## Repository Structure

- **/code:**  
  Contains implementation of the score matching framework including data preprocessing, model definition, adaptive hyperparameter tuning, and training scripts.
  
- **/experiments:**  
  Scripts and notebooks for running experiments on Gaussian datasets, GGM recovery, and CIFAR-10 subset with induced missingness. Example configurations and hyperparameter candidate ranges are provided.

- **/reports:**  
  The full research paper and extended discussion reports detailing methodology, experimental results, and future research directions.

- **/docs:**  
  Additional documentation, including installation instructions, usage guidelines, and reproducibility notes.

---

## Key Features

- **Adaptive Prompt-Based Hyperparameter Selection:**  
  Automatically tunes parameters such as IW sample size, number of variational inner-loop steps, learning rates for score and variational parameters, and truncation parameters to optimize convergence behavior.

- **Robust Precision Matrix Estimation:**  
  Employs lower triangular parametrization to ensure positive-definiteness, along with an optional \(L_1\) penalty on off-diagonal elements for sparsity in GGMs.

- **Handling of Partial Observations:**  
  The surrogate loss function is designed to operate on incomplete data by incorporating a binary mask that indicates observed entries, ensuring robust estimation under MCAR conditions.

- **Experimental Results:**  
  - **Gaussian Experiments:** The surrogate loss decreased from 9.687 (iteration 50) to 0.094 (iteration 300) with parameter error improvement.
  - **GGM Recovery:** ROC AUC improved from 0.219 to 0.972 over the training iterations, demonstrating excellent structural recovery.

---

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Recommended package manager: `pip` or `conda`

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/adaptive-prompt-enhanced-score-matching.git
   cd adaptive-prompt-enhanced-score-matching
   ```

2. **Create and Activate a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   The requirements file includes all necessary libraries (e.g., numpy, scipy, torch, matplotlib).

### Running Experiments

The repository includes several scripts for running experiments on different datasets.

- **Gaussian Experiment:**

  ```bash
  python experiments/run_gaussian_experiment.py
  ```

- **Gaussian Graphical Model (GGM) Recovery:**

  ```bash
  python experiments/run_ggm_experiment.py
  ```

- **CIFAR-10 Partial Observation Experiment:**

  ```bash
  python experiments/run_cifar_experiment.py
  ```

Configuration files in the `/experiments/configs/` directory allow you to adjust hyperparameters and missingness rates.

---

## Detailed Documentation

For a detailed explanation of the methodology, please refer to:

- **Research Report:**  
  [Adaptive Prompt-Enhanced Score Matching for Partially Observed Data (PDF)](docs/research_report.pdf)

- **Extended Discussion & Future Research:**  
  [Extended Discussion](docs/extended_discussion.pdf)

- **API Documentation:**  
  [Code Documentation](docs/api_reference.md)

---

## Contributing

We welcome contributions and feedback! Feel free to open issues or submit pull requests with improvements to the code, documentation, or experiments.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or discussions about this project, please contact:

- Research Group: Agent Laboratory
- Email: agent.lab@example.com

---

Happy experimenting and thank you for exploring Adaptive Prompt-Enhanced Score Matching for Partially Observed Data!