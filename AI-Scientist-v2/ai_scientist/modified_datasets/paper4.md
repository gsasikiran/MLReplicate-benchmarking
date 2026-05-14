# Title: Score Matching with Missing Data

## Keywords

Score Matching, Missing Data, Importance Weighting (IW), Variational Approach, Diffusion Processes, Energy-based Modelling, Graphical Model Estimation, Missing Score Matching, Fisher Divergence, Truncated Score Matching

## TL;DR

The paper introduces methods to adapt score matching for use with missing data, addressing a gap in the existing literature. Score matching is a key technique in areas like diffusion processes and energy-based modeling. The authors propose two main variations to handle data that is partially missing over any subset of coordinates:

    Importance Weighting (IW) Approach: This method is shown to be effective in smaller sample, lower-dimensional cases and allows for finite sample bounds.

Variational Approach: This is demonstrated to be stronger in more complex, high-dimensional settings, which the authors show on graphical model estimation tasks.

The overall contribution is a flexible score matching framework compatible with any parameterized score model to learn the full score function from partially missing multi-dimensional input data, a paradigm they term "missing score matching".

## Abstract

Score matching is a vital tool for learning the dis-
tribution of data with applications across many
areas including diffusion processes, energy based
modelling, and graphical model estimation. De-
spite all these applications, little work explores
its use when data is incomplete. We address this
by adapting score matching (and its major exten-
sions) to work with missing data in a flexible set-
ting where data can be partially missing over any
subset of the coordinates. We provide two sep-
arate score matching variations for general use,
an importance weighting (IW) approach, and a
variational approach. We provide finite sample
bounds for our IW approach in finite domain set-
tings and show it to have especially strong perfor-
mance in small sample lower dimensional cases.
Complementing this, we show our variational ap-
proach to be strongest in more complex high-
dimensional settings which we demonstrate on
graphical model estimation tasks on both real and
simulated data.
