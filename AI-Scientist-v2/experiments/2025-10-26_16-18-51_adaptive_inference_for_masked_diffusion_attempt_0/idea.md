## Name

adaptive_inference_for_masked_diffusion

## Title

Train for the Worst, Plan for the Best: Enhancing Token Ordering in Masked Diffusions

## Short Hypothesis

Leveraging adaptive inference strategies can significantly enhance the performance of masked diffusion models on complex generative tasks by allowing the model to sidestep computationally intractable subproblems during decoding.

## Related Work

Current literature highlights the challenges faced by masked diffusion models in handling computationally intensive tasks and the performance of autoregressive models, which excel in structured decoding. However, existing works do not fully explore adaptive inference strategies that leverage the flexibility of MDMs during inference. Our proposal aims to bridge this gap by systematically investigating and implementing adaptive token ordering strategies, which have not been addressed in the recent literature.

## Abstract

Masked diffusion models (MDMs) have emerged as a powerful paradigm for generative modeling over discrete domains. However, their training often involves solving computationally intractable problems, while their inference capabilities remain underutilized. In this work, we propose to enhance the performance of MDMs by introducing adaptive inference strategies that allow for dynamic token ordering during decoding. We demonstrate that by sidestepping computationally heavy subproblems, pretrained MDMs can achieve significant performance improvements on complex tasks such as logic puzzles. Our experiments show that adaptive inference boosts Sudoku solving accuracy from less than 7% to approximately 90%, even outperforming autoregressive models with significantly more parameters. This work opens new avenues for leveraging the strengths of MDMs in discrete generative tasks.

## Experiments

- 1. Baseline Evaluation: Compare the performance of a standard MDM and an ARM on Sudoku puzzles using traditional decoding methods.
- 2. Adaptive Inference Implementation: Develop an adaptive decoding strategy that selects the token order based on prior context during inference.
- 3. Performance Comparison: Evaluate the enhanced MDM on Sudoku and other logic puzzles, measuring solving accuracy and time taken to complete tasks against the baseline and ARMs.
- 4. Robustness Testing: Analyze the performance of the adaptive inference strategy on varying complexity levels of logic puzzles to ensure generalizability.

## Risk Factors And Limitations

- Potential overfitting to specific types of puzzles if the adaptive strategy is too narrowly focused.
- Difficulty in generalizing adaptive inference strategies across diverse discrete generative tasks.
- Computational overhead involved in implementing adaptive inference strategies may affect real-time applications.

## Code To Potentially Use

Use the following code as context for your experiments:

```python
# Auto-generated Python file for idea

```

