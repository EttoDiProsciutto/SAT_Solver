# SAT_Solver

Project Overview and Instructions

This project is developed for educational purposes and explores the application of machine learning and symbolic reasoning techniques to the task of propositional logic formula classification. Specifically, the goal is to determine whether a given well-formed propositional logic formula is satisfiable (SAT) or unsatisfiable (UNSAT).

The project provides a series of scripts that allow for the comparison of three different approaches:

    Pretrained BERT (bert-base-uncased) fine-tuned for the SAT/UNSAT classification task.

    Mini-BERT, a lightweight BERT model trained from scratch with a custom tokenizer designed specifically for propositional logic formulae.

    A procedural solver, based on resolution and CNF transformation, that deterministically evaluates the satisfiability of the formula.

Script Workflow

The scripts must be executed in the following order:

    Formula Generation Script
    This script automatically generates well-formed propositional logic formulae and splits them into training and testing datasets. The output is typically stored in CSV format.

    Fine-tuning Script for bert-base-uncased
    This script performs supervised fine-tuning of the pretrained bert-base-uncased model on the generated dataset. It uses token-level encoding with the default BERT tokenizer.

    Training Script for Mini-BERT (from scratch)
    This script trains a compact BERT architecture initialized from scratch. It includes the construction of a minimal vocabulary tailored to propositional logic and uses a custom tokenizer.

    Model Evaluation Script
    This script tests the accuracy, precision, recall, and F1-score of the fine-tuned bert-base-uncased model and the Mini-BERT model on the test dataset.

    CNF Conversion Script
    Since the procedural solver operates on formulas in Conjunctive Normal Form (CNF), this script is used to convert well-formed formulas into CNF equivalents. It is a prerequisite for the procedural solver.

    Procedural Solver Script
    This script takes CNF formulas as input and applies a resolution-based approach to determine their satisfiability. It provides a symbolic, rule-based baseline for comparison against the learned models.

Disclaimer

Parts of the codebase were generated with the assistance of generative AI, specifically OpenAI's GPT-4.1 model (ChatGPT). However, I have reviewed, verified, and revised all code to ensure correctness and clarity. I take full responsibility for the final implementation and any decisions made in the course of this project's development.
