# Calibrated Satellite Image Classification with BNNs

## Task Overview

This task involves implementing a **Bayesian Neural Network (BNN)** to classify 60x60 satellite images of land usage. The primary goal is to create a **well-calibrated** model that can accurately report its uncertainty. This is crucial because the test set contains ambiguous images (e.g., mixed land types, clouds, snow) not present in the clean training set.

## Implementation

Your objective is to implement the **SWA-Gaussian (SWAG)** algorithm, a method for approximate Bayesian inference, to build the BNN. All your code should be added to `solution.py`.

* **Part 1:** Implement **SWAG-Diagonal** using the provided MAP-trained weights (`map_weights.pt`) as a starting point.
* **Part 2:** Implement **full SWAG** and work to improve your model's calibration and reduce the overall cost.

## Evaluation

For each test image, your model must output either a specific class (0-5) or "don't know" (-1).

* **Prediction Cost:** A custom cost function is applied:
    * **Correct label:** 0 cost
    * **"Don't know":** 1 cost (fixed)
    * **Incorrect label:** 3 cost (high penalty)
    * **Any label for an ambiguous sample:** 3 cost (high penalty)

* **Calibration Penalty:** The model's **Expected Calibration Error (ECE)** is measured. Any ECE value above 0.1 is added as a penalty to the final score.

* **Goal:** Achieve a total cost (prediction cost + ECE penalty) below **0.856**.
