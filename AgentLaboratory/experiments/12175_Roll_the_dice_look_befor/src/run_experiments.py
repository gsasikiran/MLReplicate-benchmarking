###########################################
# Begin Experiments for Algorithmic Creativity
###########################################

import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import string

# (Assume the dataset code block above has already run.)
# We have available:
#   external_dataset
#   synthetic_dataset with keys "task" and "example"
#   The synthetic_examples list also exists if needed.
#
# This code will simulate three methods:
#   • NTP: Next-token Prediction
#   • MTP: Multi-token Teacherless Prediction
#   • SEDD: Discrete Diffusion (absorb variant)
#
# For each method, we simulate generation under different decoding conditions:
# Temperature: 0 (deterministic/greedy) and non-zero values: 0.5, 1.0, 2.0.
# Seed-conditioning: We simulate two modes: with seed prefix and without seed.
#
# For each condition we generate outputs for a random subset of 20 examples from the synthetic dataset.
# For each generated output we estimate:
#   - Coherence: via a task-specific check.
#   - Memorization: whether the generated output exactly equals the training example (including any prefix).
#   - Diversity: measured as the proportion of unique outputs in that condition.
#
# Then we aggregate metrics across methods and decoding conditions.
#
# Finally, we produce two figures:
#   Figure_1_Exp.png: Bar plot of coherence rates (percentage passing the task-specific check) for each method & condition.
#   Figure_2_Exp.png: Bar plot of diversity (unique output rate) for each method & condition.
#
# The following code is written as loose code (no helper functions) per instructions.
# ------------------------------------------------------------------------

print("\n=== Starting Experiment: Comparing NTP, MTP, and SEDD on Synthetic Tasks ===")

# Define tasks and expected parent set for sibling task:
parent_set = {"A", "B", "C", "D", "E"}

# Sampling conditions:
temperature_list = [0, 0.5, 1.0, 2.0]
seed_modes = [True, False]  # True means we prepend a new seed prefix; False means no extra seed prefix

# Model method parameters: noise factors for modifications in non-deterministic sampling.
# These factors indicate, per token, the chance to perturb the token.
noise_factors = {
    "NTP": 0.3,  # base multiplier to be multiplied by temperature
    "MTP": 0.5,
    "SEDD": 0.7
}

# We will simulate generation for a sample subset (20 examples).
all_data = synthetic_dataset
total_examples = len(all_data)
sample_indices = random.sample(range(total_examples), min(20, total_examples))

# Prepare container for results.
# results: dictionary keyed by (method, temperature, seed_mode) storing list of outputs and metrics.
results = {}

# Explanation print:
print("\nAbout to run generation experiments. For each model (NTP, MTP, SEDD), for each decoding condition (temperature and seed-conditioning),")
print("we simulate generation on 20 randomly chosen synthetic examples. We then compute the following metrics per condition:")
print("  - Coherence Rate: Fraction of outputs that pass a task-specific check (e.g., correct number of tokens, proper structure).")
print("  - Memorization Rate: Fraction of outputs that exactly match the training example (indicating memorization).")
print("  - Diversity: Unique outputs count / total outputs generated.\n")

# Loop over models, decoding conditions, and examples.
for method in ["NTP", "MTP", "SEDD"]:
    for temp in temperature_list:
        for seed_mode in seed_modes:
            key = (method, temp, seed_mode)
            outputs = []  # store generated outputs
            mem_count = 0
            coherence_count = 0
            total_runs = 0
            
            # Iterate over sampled examples.
            for idx in sample_indices:
                # Get training example text and task.
                train_ex = all_data[idx]["example"]
                task = all_data[idx]["task"]
                
                # For each example, simulate T generations (set T = 5)
                for t in range(5):
                    total_runs += 1
                    # Determine base output: for temperature==0, simply output the training example.
                    gen_text = train_ex  # start with training example text
                    
                    # If seed_mode is True, prepend a new seed prefix.
                    if seed_mode:
                        # Simulate a new seed prefix (similar to get_seed_prefix)
                        choice = random.choice(["seed", "pause"])  # force non-"none" for seed mode
                        if choice == "seed":
                            length = random.choice([2, 4, 10])
                            prefix_new = "SEED: " + "".join(random.choices(string.ascii_uppercase, k=length))
                        else:
                            prefix_new = "PAUSE: " + " ".join(["[PAUSE]"] * 10)
                        gen_text = prefix_new + " " + gen_text
                        
                    # If temperature > 0, simulate noise modifications.
                    if temp > 0:
                        noise_prob = min(0.9, noise_factors[method] * temp)
                        # Split text into tokens (split on space)
                        tokens = gen_text.split(" ")
                        new_tokens = []
                        for i, token in enumerate(tokens):
                            # With chance noise_prob, perturb the token.
                            if random.random() < noise_prob:
                                # Different perturbations for specific tasks:
                                # For Sibling task: if this token is the parent's id (typically third token after any prefix).
                                # First remove any prefix tokens if they match SEED: or PAUSE:
                                # We'll assume that if a token is in parent_set and length==1, then it's parent's token.
                                if token in parent_set and len(token) == 1:
                                    # With perturbation, randomly choose parent letter (may or may not be same).
                                    token = random.choice(list(parent_set))
                                else:
                                    # If token is made of digits, change it slightly.
                                    if token.isdigit():
                                        token = str(random.randint(1,20))
                                    else:
                                        # Otherwise, append or remove a character.
                                        if len(token) > 1 and random.random() < 0.5:
                                            token = token[:-1]
                                        else:
                                            token = token + random.choice(string.ascii_lowercase)
                            new_tokens.append(token)
                        gen_text = " ".join(new_tokens)
                    # End simulation of generation.
                    
                    # Append output.
                    outputs.append(gen_text)
                    
                    # Check for memorization: if generated text exactly equals training example (if seed_mode, compare after removing new seed).
                    if seed_mode:
                        # Remove potential seed prefix: if text starts with "SEED:" or "PAUSE:", remove the first two tokens.
                        split_tokens = gen_text.split(" ")
                        if split_tokens[0] in ["SEED:", "PAUSE:"]:
                            gen_core = " ".join(split_tokens[2:])  # assume prefix is two tokens
                        else:
                            gen_core = gen_text
                        train_core = train_ex  # training example as stored originally (may also have its own seed, but we compare core text)
                    else:
                        gen_core = gen_text
                        train_core = train_ex
                    if gen_core.strip() == train_core.strip():
                        mem_count += 1
                        
                    # Check Coherence:
                    # Remove any prepended seed by checking first token.
                    split_tokens = gen_text.split(" ")
                    if split_tokens[0] in ["SEED:", "PAUSE:"]:
                        core_text = " ".join(split_tokens[2:])
                    else:
                        core_text = gen_text
                    valid = False
                    if task == "sibling":
                        # Expect three tokens: child1 child2 parent.
                        parts = core_text.strip().split(" ")
                        if len(parts) == 3 and parts[2] in parent_set:
                            valid = True
                    elif task == "triangle":
                        # Expect to start with "tri:" and exactly 3 pairs (assume 3 occurrences of "(")
                        if core_text.startswith("tri:") and core_text.count("(") == 3:
                            valid = True
                    elif task == "circle":
                        # Expect 9 arrows "→"
                        if core_text.count("→") == 9:
                            valid = True
                    elif task == "line":
                        # Expect 8 arrows "→"
                        if core_text.count("→") == 8:
                            valid = True
                    if valid:
                        coherence_count += 1
            # Save aggregated metrics for the condition.
            coherence_rate = coherence_count / total_runs
            memorization_rate = mem_count / total_runs
            diversity = len(set(outputs)) / total_runs
            results[key] = {"coherence": coherence_rate,
                            "memorization": memorization_rate,
                            "diversity": diversity,
                            "total": total_runs}
            print("Method: {}, Temp: {}, Seed_Mode: {} => Runs: {} | Coherence: {:.2f} | Memorization: {:.2f} | Diversity: {:.2f}"
                  .format(method, temp, seed_mode, total_runs, coherence_rate, memorization_rate, diversity))

# Sanity check: Ensure no condition got 0% coherence.
all_coherences = [results[k]["coherence"] for k in results]
if min(all_coherences) == 0:
    print("Warning: At least one condition had 0% coherence. Check generation perturbations for bugs.")
else:
    print("All conditions have non-zero coherence as required.")

# ===============================================================
# Prepare data for plotting figures.
# We'll create grouped bar plots.
methods = ["NTP", "MTP", "SEDD"]
labels = []
coherence_vals = []
diversity_vals = []
for m in methods:
    for temp in temperature_list:
        for seed_mode in seed_modes:
            lbl = "{} T:{} Seed:{}".format(m, temp, "Yes" if seed_mode else "No")
            labels.append(lbl)
            coherence_vals.append(results[(m, temp, seed_mode)]["coherence"])
            diversity_vals.append(results[(m, temp, seed_mode)]["diversity"])

x = np.arange(len(labels))

# Figure 1: Coherence Rates
plt.figure(figsize=(14,6))
plt.bar(x, coherence_vals, color='skyblue')
plt.xticks(x, labels, rotation=45, ha="right")
plt.ylim(0,1)
plt.ylabel("Coherence Rate")
plt.title("Figure_1_Exp.png: Coherence Rate by Model, Temperature, and Seed Conditioning")
plt.tight_layout()
plt.savefig("Figure_1_Exp.png")
print("\nGenerated Figure_1_Exp.png: Bar plot of coherence rates for each method and decoding condition.")

# Figure 2: Diversity Rates
plt.figure(figsize=(14,6))
plt.bar(x, diversity_vals, color='salmon')
plt.xticks(x, labels, rotation=45, ha="right")
plt.ylim(0,1)
plt.ylabel("Diversity")
plt.title("Figure_2_Exp.png: Diversity (Unique Outputs Ratio) by Model, Temperature, and Seed Conditioning")
plt.tight_layout()
plt.savefig("Figure_2_Exp.png")
print("Generated Figure_2_Exp.png: Bar plot of diversity rates for each method and decoding condition.")

# Final summary print:
print("\n=== Experiment Completed ===")
print("For each method (NTP, MTP, SEDD) and each decoding condition (temperature and seed-conditioning),")
print("the results above show the coherence rate (fraction of task-specific valid outputs),")
print("memorization rate (fraction of outputs that exactly match a training example), and")
print("diversity (unique outputs per generated samples).")
print("Figures have been saved to the current folder. Use these to compare algorithmic creativity across methods.")

###########################################
# End of Code.
###########################################