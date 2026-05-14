# Title: The Value of Prediction in Identifying the Worst-Off

## Keywords
Core Concepts, Policy/Context, Applications/Case Study, Modeling

## TL;DR

This paper examines the relative value of improving predictive accuracy (e.g., better machine learning models) versus expanding screening capacity (e.g., hiring more staff, increasing budget) in government programs designed to identify and help the "worst-off" (most vulnerable) individuals, such as those at risk of long-term unemployment or poverty.

Key Findings (TLDR)

    Prediction is a "First and Last-Mile Effort": Improvements in predictive accuracy (R2) have the highest marginal impact only when the model is either very poor (R2→0) or nearly perfect (R2→1 and screening capacity α matches the target population β).

Expanding Capacity is Usually More Effective: For the typical operating regime of most allocation programs—where predictions are moderate (e.g., R2 is not too extreme) and capacity is constrained (α≤β)—expanding screening capacity yields a significantly greater increase in the ability to identify the worst-off compared to small improvements in prediction accuracy.

The Prediction-Access Ratio (PAR): The paper introduces the PAR to quantify this trade-off: PAR=Marginal Value of Better PredictionMarginal Value of Expanding Access​. Policymakers should expand access if the cost of access divided by the cost of prediction is less than the PAR (CAccess​/CPred​<PAR), and invest in better prediction otherwise.Scarce Capacity: When screening resources are severely limited (α≪β), the benefit of expanding screening capacity is overwhelming, leading to a very large PAR.

Real-World Validation: A case study on long-term unemployment amongst German jobseekers validates the theoretical findings, showing that in a real-world context, expanding screening capacity generally has a greater impact than enhancing prediction accuracy.

## Abstract

Machine learning is increasingly used in govern-
ment programs to identify and support the most
vulnerable individuals, prioritizing assistance for
those at greatest risk over optimizing aggregate
outcomes. This paper examines the welfare im-
pacts of prediction in equity-driven contexts, and
how they compare to other policy levers, such as
expanding bureaucratic capacity. Through math-
ematical models and a real-world case study on
long-term unemployment amongst German resi-
dents, we develop a comprehensive understanding
of the relative effectiveness of prediction in sur-
facing the worst-off. Our findings provide clear
analytical frameworks and practical, data-driven
tools that empower policymakers to make princi-
pled decisions when designing these systems.
