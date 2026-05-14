# Adaptive Inference in Masked Diffusion Models

Welcome to the GitHub repository for our project on **Adaptive Inference in Masked Diffusion Models (MDMs)**. This repository contains the code, experimental setups, and detailed reports associated with our research on dynamically selecting token-ordering strategies to improve the performance of discrete generative models. The project is inspired by and builds upon recent advances in masked diffusion and autoregressive modeling.

---

## Overview

Masked diffusion models have emerged as a promising framework for discrete generative tasks, particularly in text generation and structured puzzle solving. In our work, we address two major challenges:

1. **Exponential Infilling Complexity:** MDMs are trained on an exponential number of reconstruction subproblems, many of which are computationally intractable.
2. **Order-Agnostic Uncertainty:** The flexibility to predict tokens in arbitrary orders during inference introduces uncertainty that degrades performance (e.g., higher negative log likelihood (NLL) and perplexity).

To overcome these issues, our adaptive inference framework:
- Uses a **margin-based criterion** (based on the difference between the highest and second-highest token probabilities) to determine prediction confidence.
- Dynamically selects between **deterministic token selection** (when confidence is high) and **Top-K sampling** (when prediction uncertainty is high).
- Achieves significant improvements in NLL, perplexity, and puzzle solving accuracy while maintaining token diversity.

---

## Key Features

- **Adaptive Inference Strategies:**  
  - **Top-K Sampling:** Samples from the top K tokens (default K = 5) when the confidence margin is low.
  - **Top-K Margin Method:** Combines margin thresholding (default τ = 0.3) with Top-K sampling to bypass difficult subproblems.

- **Empirical Validation:**  
  - Text generation experiments show a reduction in average NLL from 6.8510 to 4.4384 and perplexity from 944.82 to 84.64 using adaptive strategies.
  - Synthetic puzzle experiments (L&O-NAE-SAT style) see solve rates improve from 50.03% (vanilla inference) to 84.95% (Top-K Margin).

- **Scalability Analysis:**  
  - π-Learner scaling law experiments demonstrate consistent improvements in validation loss with increased model capacity across different token permutation regimes.

- **Implementation:**  
  - A simplified Transformer-based masked diffusion model.
  - Configurable hyperparameters including margin threshold, Top-K value, temperature, reverse diffusion steps, and more.
  
---

## Repository Structure

```
Adaptive-Inference-MDM/
├── README.md              # This file
├── LICENSE
├── docs/
│   └── full_report.pdf    # Complete research paper with discussion and experimental results
├── experiments/
│   ├── text_generation/   # Code and data for text generation experiments
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── config.yaml
│   ├── puzzle_solving/    # Code for synthetic puzzle (L&O-NAE-SAT) experiments
│   │   ├── train_puzzles.py
│   │   └── evaluate.py
│   └── scaling/           # π-Learner and scaling law experiments
│       ├── scaling_experiments.py
│       └── config_scaling.yaml
├── models/
│   └── masked_diffusion_model.py   # Implementation of the Transformer-based MDM
├── utils/
│   ├── tokenizer.py       # DummyTokenizer for tokenizing sequences
│   └── helpers.py         # Utility functions (e.g., for Top-K sampling, margin computation)
└── requirements.txt       # Python package dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.8 or later
- [PyTorch](https://pytorch.org/) (for model implementation and training)
- Other required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Adaptive-Inference-MDM.git
   cd Adaptive-Inference-MDM
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training

- **Text Generation Training:**  
  Execute the training script in the `experiments/text_generation` folder:
  ```bash
  python experiments/text_generation/train.py --config experiments/text_generation/config.yaml
  ```

- **Puzzle Solving Training:**  
  Use the dedicated script for training/evaluation in the `experiments/puzzle_solving` folder:
  ```bash
  python experiments/puzzle_solving/train_puzzles.py --config experiments/puzzle_solving/config.yaml
  ```

### Inference

- **Adaptive Inference for Text Generation:**  
  Run the inference script that implements the adaptive token ordering:
  ```bash
  python experiments/text_generation/inference.py --config experiments/text_generation/config.yaml
  ```

- The inference script includes both vanilla and adaptive (Top-K, Top-K Margin) strategies. Adjust hyperparameters such as margin threshold (τ) and Top-K value via the configuration file.

### Scaling Experiments

- To replicate the π-Learner scaling law experiments:
  ```bash
  python experiments/scaling/scaling_experiments.py --config experiments/scaling/config_scaling.yaml
  ```

---

## Experimental Results

The following summarizes our main empirical findings:

### Text Generation

| Method                 | NLL    | Perplexity |
|------------------------|--------|------------|
| Vanilla MDM            | 6.8510 | 944.82     |
| Top-K Sampling         | 4.4384 | 84.64      |
| Top-K Margin           | 5.5970 | 269.63     |

Token-frequency entropy remains nearly constant (~3.88) across methods.

### Puzzle Solving

| Inference Strategy     | Solve Rate (%) |
|------------------------|----------------|
| Vanilla                | 50.03          |
| Top-K Sampling         | 70.24          |
| Top-K Margin           | 84.95          |

### Scaling Laws (π-Learner Experiments)

Under various permutation regimes (e.g., pi_unif, pi_closer), validation NLL improved systematically with increased model capacity:
- Example improvement under pi_unif: 2.8468 → 2.7145

Detailed plots and figures (e.g., `Figure_1_TextAdaptive.png` and `Figure_2_PuzzleAccuracy.png`) are provided in the `docs/` folder.

---

## Citation

If you find this work useful in your research, please consider citing our paper:

```
@misc{adaptive_inference_mdm,
  title={Adaptive Inference in Masked Diffusion Models},
  author={Agent Laboratory},
  year={2023},
  eprint={XXXX.XXXX},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions to enhance adaptive inference strategies, experiments, and overall code quality are welcome. Please fork the repository and submit a pull request with your proposed changes.

For major changes, please open an issue first to discuss what you would like to change.

---

## Contact

For questions or inquiries, please contact [your.email@university.edu](mailto:your.email@university.edu).

Happy Generating!  
— The Agent Laboratory Research Team

--- 

This README integrates all relevant aspects of our research, including theoretical background, system architecture, experimental details, and usage instructions. We hope it serves as a comprehensive starting point for researchers and practitioners interested in advancing generative modeling with adaptive inference. Enjoy exploring the code and experiments!