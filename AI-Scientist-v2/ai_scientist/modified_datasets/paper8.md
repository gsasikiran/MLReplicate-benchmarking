# Title: COLLAB LLM: From Passive Responders to Active Collaborators

## Keywords
COLLABLLM, Large Language Models (LLMs), Multiturn Collaboration, Multiturn-aware Rewards (MR), Collaborative Simulation, Reinforcement Learning (RL) finetuning, User Intent Discovery, Conversational Efficiency, Interactivity

## TL;DR

The paper introduces COLLABLLM, a novel training framework that transforms Large Language Models (LLMs) from passive responders to active collaborators in multi-turn interactions.

Traditional LLMs are trained with next-turn rewards, leading to passive responses and inefficient conversations when users have ambiguous requests or long-term goals. COLLABLLM solves this by using a collaborative simulation to estimate the long-term impact of a response across multiple turns, which is quantified as a Multiturn-aware Reward (MR). Reinforcement fine-tuning with MRs encourages the LLM to actively uncover user intent and offer insightful suggestions.

COLLABLLM significantly outperforms existing baselines, achieving an average of 18.5% higher task performance and 46.3% enhanced interactivity across three challenging multi-turn benchmarks. A large user study also showed a 17.6% increase in user satisfaction and a 10.4% reduction in user spent time.

## Abstract

Large Language Models are typically trained
with next-turn rewards, limiting their ability to
optimize for long-term interaction. As a result,
they often respond passively to ambiguous or
open-ended user requests, failing to help users
reach their ultimate intents and leading to inef-
ficient conversations. To address these limita-
tions, we introduce C OLLAB LLM, a novel and
general training framework that enhances mul-
titurn human-LLM collaboration. Its key in-
novation is a collaborative simulation that es-
timates the long-term contribution of responses
using Multiturn-aware Rewards. By reinforce-
ment fine-tuning these rewards, C OLLAB LLM
goes beyond responding to user requests, and ac-
tively uncovers user intent and offers insightful
suggestions—a key step towards more human-
centered AI. We also devise a multiturn interac-
tion benchmark with three challenging tasks such
as document creation. C OLLAB LLM signifi-
cantly outperforms our baselines with averages
of 18.5% higher task performance and 46.3% improved interactivity by LLM judges. Finally, we conduct
a large user study with 201 judges, where C OLLAB LLM
increases user satisfaction by 17.6% and reduces user spent
time by 10.4%
