#!/usr/bin/env python3
"""Run ONLY Paper Generation Phase"""

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

# Set environment variables
os.environ['CATEGORY'] = 'vq'
os.environ['INSTANCE_ID'] = 'bayesian_quadrature'
os.environ['COMPLETION_MODEL'] = 'gpt-4o'

from run_ai_researcher import run_ai_researcher
from directory_manager import get_directory_manager

if __name__ == "__main__":
    print("="*70)
    print("AI-RESEARCHER: PAPER GENERATION ONLY")
    print("="*70)
    print(f"Category: {os.environ['CATEGORY']}")
    print(f"Instance: {os.environ['INSTANCE_ID']}")
    
    manager = get_directory_manager()
    
    # Specify which research run to use
    project_root = Path(__file__).parent
    research_run = project_root / "results" / "test_016"  # Change this to your completed run
    
    if research_run.exists():
        print(f"\n✓ Using research from: {research_run}")
        manager.current_run_dir = research_run
    else:
        print(f"\n❌ Research run not found: {research_run}")
        print("\nAvailable runs:")
        for run in sorted((project_root / "results").glob("test_*")):
            has_cache = (run / "cache").exists()
            has_workplace = (run / "workplace").exists()
            status = "✓" if has_cache and has_workplace else "✗"
            print(f"  {status} {run.name}")
        exit(1)
    
    print("\n📝 Starting Paper Generation...")
    
    paper_result = run_ai_researcher(
        "Conformal Prediction as Bayesian Quadrature. Reinterprets conformal prediction through a Bayesian lens, proposing a Bayesian quadrature framework that provides interpretable and flexible uncertainty guarantees. This approach extends traditional conformal prediction by integrating prior knowledge into uncertainty estimation, yielding richer and more informative loss distributions for high-stakes machine learning applications.",
        "Aitchison, J. and Dunsmore, I. R. Statistical Prediction Analysis, Vovk, V., Gammerman, A., and Shafer, G. Algorithmic Learning in a Random World, Angelopoulos, A. N. and Bates, S. Conformal prediction: A gentle introduction, Cockayne, J., Oates, C. J., Sullivan, T. J., and Girolami, M. Bayesian probabilistic numerical methods, Diaconis, P. Bayesian numerical analysis, Hennig, P., Osborne, M. A., and Kersting, H. P. Probabilistic Numerics: Computation as Machine Learning, Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., and Schuster, T. Conformal risk control, Papadopoulos, H., Proedrou, K., Vovk, V., and Gammerman, A. Inductive confidence machines for regression, O'Hagan, A. Bayes-Hermite quadrature, Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C. L. Microsoft COCO: Common objects in context, Shafer, G. and Vovk, V. A tutorial on conformal prediction, Barber, R. F., Candès, E. J., Ramdas, A., and Tibshirani, R. J. The limits of distribution-free conditional predictive inference, Gibbs, I., Cherian, J. J., and Candès, E. J. Conformal Prediction With Conditional Guarantees, Ng, K. W., Tian, G.-L., and Tang, M.-L. Dirichlet and Related Distributions: Theory, Methods and Applications, Tukey, J. W. Nonparametric estimation II. Statistically equivalent blocks and tolerance regions—the continuous case.",
        "Paper Generation Agent"
    )

    print(f"\n✓ Complete!")
    print(f"Status: {paper_result[2]}")