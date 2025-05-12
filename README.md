# Exchange Rate Prediction with PSO-Optimized Neural Networks

This project implements and compares neural network architectures for predicting the GBP/USD exchange rate. It features a custom Particle Swarm Optimization (PSO) algorithm for hyperparameter tuning, resulting in significant improvements over baseline models.

---

## 1. Project Overview

- **Goal:** Predict GBP/USD exchange rates using neural networks (MLP and CNN), optimized via PSO.
- **Key Features:**
  - Custom PSO for hyperparameter optimization
  - Univariate and multivariate time series forecasting
  - Comparison of baseline and PSO-optimized models
  - Implementation of Swish, ReLU, and Tanh activation functions

---

## 2. Directory Structure

```
PSO/
│
├── DATA/                # Raw and processed data files
├── PLOTS/               # Generated plots and figures
├── 2. Data Preparation.ipynb
├── 3. Base Model.ipynb
├── 4. MLP Univariate.ipynb
├── 5. CNN Univariate.ipynb
├── 9. PSO MLP Univariate.ipynb
├── 10. PSO CNN Univariate.ipynb
├── model_comparison_plot.py
├── pso_optimizer.py
├── README.md
└── Technical Report.docx
```

---

## 3. Data Sources

1. **Normalized GDP (monthly):** Federal Reserve Economic Data (FRED)
2. **Libor Rates (daily):** FRED
3. **Current Account to GDP (quarterly):** OECD Statistics
4. **Forex (daily):** Federal Reserve

---

## 4. Requirements

- Python 3.11+
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

**Install dependencies:**
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

## 5. Step-by-Step Usage

### 5.1. Data Preparation

1. **Open** `2. Data Preparation.ipynb`.
2. **Run all cells** to:
   - Download and merge data from all sources
   - Normalize and preprocess the data
   - Save the processed dataset to the `DATA/` directory

### 5.2. Baseline Model Training

3. **MLP Univariate:**
   - Open `4. MLP Univariate.ipynb`
   - Train a basic MLP model on univariate data
   - Record the RMSE for comparison

4. **CNN Univariate:**
   - Open `5. CNN Univariate.ipynb`
   - Train a basic CNN model on univariate data
   - Record the RMSE for comparison

### 5.3. PSO-Optimized Model Training

5. **PSO MLP Univariate:**
   - Open `9. PSO MLP Univariate.ipynb`
   - Use the custom PSO optimizer (`pso_optimizer.py`) to search for optimal MLP hyperparameters
   - Train the best-found MLP model and evaluate performance

6. **PSO CNN Univariate:**
   - Open `10. PSO CNN Univariate.ipynb`
   - Use the PSO optimizer to search for optimal CNN hyperparameters
   - Train the best-found CNN model and evaluate performance

### 5.4. Results Visualization

7. **Model Comparison:**
   - Use `model_comparison_plot.py` to generate plots comparing the performance of all models
   - View plots in the `PLOTS/` directory

---

## 6. PSO Hyperparameter Optimization Details

- **What is PSO?**  
  Particle Swarm Optimization is a population-based optimization algorithm inspired by the social behavior of birds. Each "particle" represents a candidate solution (set of hyperparameters) and moves through the search space to find the best solution.

- **How is it used here?**  
  The `pso_optimizer.py` module defines:
  - The `Particle` class (a candidate solution)
  - The `PSOOptimizer` class (manages the swarm and optimization process)
  - Helper functions to build MLP and CNN models from hyperparameters

- **Optimization Steps:**
  1. Define the hyperparameter search space (bounds)
  2. Initialize a swarm of particles with random hyperparameters
  3. Evaluate each particle (train and validate a model)
  4. Update velocities and positions based on personal and global bests
  5. Repeat for a set number of iterations
  6. Return the best hyperparameters found

---

## 7. Results

| Model                    | RMSE (Lower is Better) | Improvement (%) |
|--------------------------|-----------------------|----------------|
| MLP Univariate (Baseline)| 0.01295               | -              |
| **PSO-MLP Univariate**   | **0.00708**           | **45.33%**     |
| CNN Univariate (Baseline)| 0.03028               | -              |
| **PSO-CNN Univariate**   | **0.00654**           | **78.40%**     |

- **Conclusion:**  
  PSO-optimized models significantly outperform their baseline counterparts.

---

## 8. File Descriptions

- **pso_optimizer.py**:  
  Core PSO implementation and model-building utilities.

- **2. Data Preparation.ipynb**:  
  Data loading, cleaning, and merging.

- **4. MLP Univariate.ipynb**:  
  Baseline MLP model training and evaluation.

- **5. CNN Univariate.ipynb**:  
  Baseline CNN model training and evaluation.

- **9. PSO MLP Univariate.ipynb**:  
  PSO-optimized MLP model training.

- **10. PSO CNN Univariate.ipynb**:  
  PSO-optimized CNN model training.

- **model_comparison_plot.py**:  
  Script for visualizing and comparing model results.

- **PLOTS/**:  
  Output plots and figures.

- **DATA/**:  
  Input and processed data files.

---

## 9. How to Extend

- Add new data sources to `2. Data Preparation.ipynb`
- Modify or add new model architectures in the respective notebooks
- Adjust PSO search space in the PSO notebooks or `pso_optimizer.py`
- Use the PSO optimizer for other time series or regression tasks

---

## 10. References

- [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/)
- [OECD Statistics](https://stats.oecd.org/)
- [Particle Swarm Optimization - Wikipedia](https://en.wikipedia.org/wiki/Particle_swarm_optimization)

---

## 11. Contact

For questions or contributions, please contact the project maintainer.

---