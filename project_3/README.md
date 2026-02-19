# PAI25 Project - Task 3
**Group:** Volta (gguidarini, mdenegri, mmeciani)

## 🧬 Task Overview

This task involves using **Bayesian optimization** to find the optimal "structural features" ($x$) of a drug candidate. The goal is to find a candidate $x$ that:

1.  **Maximizes Bioavailability:** Achieves the highest possible logP value, represented by a noisy, expensive-to-evaluate objective function $f(x)$.
2.  **Satisfies Synthesizability:** Remains "easy to synthesize," defined by a constraint on its synthetic accessibility (SA) score, $v(x) < \kappa$, where $\kappa = 4$.

The core challenge is to find $\text{argmax}_{x \in [0, 10], v(x) < \kappa} f(x)$ given that both $f(x)$ and $v(x)$ are unknown black-box functions accessible only through noisy evaluations.

## Implementation

Your objective is to implement a **constrained Bayesian optimization** algorithm within the `BO_algo` class in the `solution.py` file.

* You must implement all methods marked with `# TODO: ...`.
* A standard Bayesian optimization algorithm (i.e., one that ignores the constraint $v(x) < \kappa$) is likely **insufficient** to pass the baseline.
* Your algorithm must effectively manage the trade-off between exploring the objective $f(x)$ and satisfying the constraint $v(x)$, especially given the limited budget for "unsafe" evaluations (where $v(x) \ge \kappa$).

## ⚖️ Evaluation

Your algorithm will be evaluated over 100 randomly generated drug discovery tasks. The final score $\bar{S}$ is designed to reward a balance of performance, safety, and non-triviality.

Your score for each task is penalized based on three factors:

1.  **Performance Regret:** How far your suggested $f(\tilde{x})$ is from the true optimal safe value.
2.  **Unsafe Evaluations:** A penalty is applied for each evaluation $x_i$ that violates the constraint $v(x_i) \ge \kappa$.
3.  **Trivial Solutions:** A penalty is applied if your final solution $\tilde{x}$ is too close to the provided initial safe point.

**Goal:** Achieve a mean score $\bar{S}$ (averaged over all 100 tasks) that is above the baseline of **0.785**.
