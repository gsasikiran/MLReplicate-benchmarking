## Name

creative_limits_lms

## Title

Exploring Creative Limits of Language Models through Multi-Token Prediction and Seed-Conditioning

## Short Hypothesis

This research investigates whether multi-token prediction and seed-conditioning enhance the algorithmic creativity of language models by allowing them to perform open-ended tasks requiring a leap of thought, compared to traditional next-token prediction.

## Related Work

Existing studies have shown that multi-token prediction (MTP) can enhance language models' performance in various tasks. However, little work has focused on quantifying the creative limits of LLMs specifically in open-ended tasks that require abstract reasoning and planning. This proposal distinguishes itself by introducing a novel test-bed of tasks and the concept of seed-conditioning, which has not been thoroughly explored in this context.

## Abstract

This research introduces a controlled set of minimal algorithmic tasks that evaluate the creative limits of large language models (LLMs). These tasks require a stochastic planning step that either discovers novel connections in knowledge graphs or constructs new patterns, simulating open-ended real-world challenges. We propose that traditional next-token learning is myopic, whereas multi-token prediction (MTP) approaches, such as teacherless training and diffusion models, excel in producing diverse and original outputs. Our novel seed-conditioning technique, which introduces randomness at the input layer, is presented as an effective method to elicit creativity without sacrificing coherence, performing comparably to existing output-layer temperature sampling. This study aims to provide a principled framework for assessing the creative capabilities of LLMs and advocates for a shift away from conventional next-token learning paradigms.

## Experiments

- Design a series of algorithmic tasks that require creative leaps, such as generating analogies, solving complex problems, or designing new mathematical challenges.
- Compare outputs from LLMs trained with traditional next-token prediction against those trained with multi-token prediction and seed-conditioning.
- Evaluate outputs using metrics for diversity and originality, such as N-gram diversity, novelty scores, and human evaluation of creativity.

## Risk Factors And Limitations

Potential risks include the challenge of effectively quantifying creativity and originality, as these concepts can be subjective. Additionally, the generalizability of findings to broader real-world tasks may be limited. There is also the risk that the introduced randomness may not lead to significant improvements in all contexts.

## Code To Potentially Use

Use the following code as context for your experiments:

```python
# Auto-generated Python file for idea

```

