# Probabilistic AI Projects

Welcome to the **Probabilistic Artificial Intelligence** repository! This collection showcases advanced machine learning projects that leverage probabilistic methods to tackle real-world challenges in prediction, classification, and optimization.

## Team Volta

- [@mmeciani](https://github.com/mmeciani)
- [@mdenegri](https://github.com/mdenegri)
- [@gguidarini](https://github.com/gguidarini)

---

## Projects Overview

### Project 1: Pollution Concentration Prediction

**Gaussian Processes for Spatial Modeling**

Predict pollution concentration levels across a 2D map using Gaussian Process Regression with an asymmetric cost function that penalizes under-prediction in residential areas.

- **Method:** Gaussian Process Regressor with Matern kernel and k-Means clustering
- **Performance:** Public Cost **4.549** (Baseline: 21.800)
- **Key Innovation:** Adaptive prediction strategy using posterior uncertainty

[View Project Details →](./project_1/)

---

### Project 2: Satellite Image Classification

**Bayesian Neural Networks for Calibrated Predictions**

Classify 60x60 satellite images of land usage using a well-calibrated Bayesian Neural Network that accurately reports uncertainty for ambiguous images.

- **Method:** SWAG (SWA-Gaussian) for approximate Bayesian inference
- **Goal:** Total cost < **0.856** (Prediction cost + ECE penalty)
- **Key Innovation:** Calibrated uncertainty estimation for "don't know" predictions

[View Project Details →](./project_2/)

---

### Project 3: Drug Discovery Optimization

**Constrained Bayesian Optimization**

Find optimal structural features of drug candidates that maximize bioavailability while satisfying synthesizability constraints using constrained Bayesian optimization.

- **Method:** Bayesian Optimization with constraint handling
- **Performance:** Mean score > **0.785** baseline
- **Key Innovation:** Balancing exploration of objective and constraint satisfaction

[View Project Details →](./project_3/)

---

## Technologies

- **Python** (NumPy, Scikit-learn, PyTorch)
- **Gaussian Processes** for spatial modeling
- **Bayesian Neural Networks** for calibrated uncertainty
- **Bayesian Optimization** for constrained optimization
- **Docker** for reproducible environments

## Repository Structure

```
.
├── project_1/          # Pollution Prediction with Gaussian Processes
├── project_2/          # Satellite Image Classification with BNNs
├── project_3/          # Drug Discovery with Bayesian Optimization
└── README.md           # This file
```

Each project directory contains its own `README.md` with detailed task overview, implementation details, and evaluation metrics.

---

## Getting Started

Navigate to any project directory and follow the instructions in the respective `README.md` file for setup and execution details.

```bash
cd project_1/  # or project_2/ or project_3/
```

---

**Course:** Probabilistic Artificial Intelligence (PAI), ETH Zurich
