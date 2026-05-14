# Title: Train for the Worst, Plan for the Best:Understanding Token Ordering in Masked Diffusions

## Keywords

Masked Diffusion Models (MDMs), Adaptive Inference, Token Ordering, Autoregressive Models (ARMs), Discrete Generative Modeling, Computational Intractability, Logic Puzzles (e.g., Sudoku), Order-Agnostic Training

## TL;DR

Masked Diffusion Models (MDMs) face a challenge during training because they must learn to solve an exponentially large number of "infilling problems," some of which are shown to be computationally intractable compared to the simpler, sequential problems solved by Autoregressive Models (ARMs).

However, this paper demonstrates that the flexibility of MDM inference—the ability to decode tokens in an arbitrary order—can be leveraged using an adaptive inference strategy to "sidestep" these difficult subproblems. Applying this adaptive approach to pretrained MDMs yields dramatic performance improvements on logic puzzles, such as boosting Sudoku solving accuracy from less than 7% to approximately 90%. This adaptive method even outperforms ARMs that were explicitly trained with teacher forcing to learn the correct decoding order.

## Abstract

In recent years, masked diffusion models (MDMs)
have emerged as a promising alternative approach
for generative modeling over discrete domains.
Compared to autoregressive models (ARMs),
MDMs trade off complexity at training time with
flexibility at inference time. At training time, they
must learn to solve an exponentially large number
of infilling problems, but at inference time, they
can decode tokens in essentially arbitrary order. In
this work, we closely examine these two compet-
ing effects. On the training front, we theoretically
and empirically demonstrate that MDMs indeed
train on computationally intractable subproblems
compared to their autoregressive counterparts. On
the inference front, we show that a suitable strat-
egy for adaptively choosing the token decoding
order significantly enhances the capabilities of
MDMs, allowing them to sidestep hard subprob-
lems. On logic puzzles like Sudoku, we show that
adaptive inference can boost solving accuracy in
pretrained MDMs from < 7% to ≈ 90%, even
outperforming ARMs with 7× as many parame-
ters and that were explicitly trained via teacher
forcing to learn the right order of decoding.
