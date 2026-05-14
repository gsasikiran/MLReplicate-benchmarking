# Title: Roll the dice & look before you leap:Going beyond the creative limits of next-token prediction

## Keywords

Next-token prediction (NTP), Multi-token prediction (MTP), Algorithmic creativity, Seed-conditioning, Teacherless training, Diffusion models, Open-ended tasks, Stochastic planning / Leap of thought

## TL;DR

This paper introduces a minimal, quantifiable test-bed of algorithmic tasks to analyze the creative limits of language models (LLMs) in open-ended, complex tasks requiring an implicit, far-sighted "leap of thought," such as designing problems or generating analogies. The authors argue that the prevailing next-token learning approach is "myopic" and shows lower "algorithmic creativity" (diversity and originality) compared to multi-token approaches, specifically teacherless training and diffusion models, which comparatively excel. Furthermore, they propose and test seed-conditioning (injecting randomness at the input layer) as an effective method to elicit diversity, finding it performs comparably to, and sometimes better than, traditional output-layer temperature sampling. The work advocates for moving beyond next-token learning and temperature sampling to improve LLMs' creative capabilities.

## Abstract

We design a suite of minimal algorithmic tasks
that are a loose abstraction of open-ended real-
world tasks. This allows us to cleanly and control-
lably quantify the creative limits of the present-
day language model. Much like real-world tasks
that require a creative, far-sighted leap of thought,
our tasks require an implicit, open-ended stochas-
tic planning step that either (a) discovers new con-
nections in an abstract knowledge graph (like in
wordplay, drawing analogies, or research) or (b)
constructs new patterns (like in designing math
problems or new proteins). In these tasks, we
empirically and conceptually argue how next-
token learning is myopic; multi-token approaches,
namely teacherless training and diffusion mod-
els, comparatively excel in producing diverse
and original output. Secondly, to elicit random-
ness without hurting coherence, we find that in-
jecting noise at the input layer (dubbed seed-
conditioning) works surprisingly as well as (and
in some conditions, better than) temperature sam-
pling from the output layer. Thus, our work offers
a principled, minimal test-bed for analyzing open-
ended creative skills, and offers new arguments
for going beyond next-token learning and temper-
ature sampling. We make part of the code avail-
able under https://github.com/chenwu98/
algorithmic-creativity
