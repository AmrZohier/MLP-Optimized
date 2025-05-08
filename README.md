# MLP Optimized: Exchange Rate Prediction

This repository contains code for predicting exchange rates using Multi-Layer Perceptron (MLP) neural networks, with optimization using Particle Swarm Optimization (PSO).

## Project Overview

The project focuses on predicting exchange rates using different neural network approaches:
- Base MLP models for univariate time series
- PSO-optimized MLP models for hyperparameter tuning
- Comparison of different approaches

## Directory Structure

- `DATA/`: Contains the dataset files
- `PLOTS/`: Output directory for generated plots and visualizations
- `*.ipynb`: Jupyter notebooks containing the implementation and experiments
- `pso_optimizer.py`: Python module for Particle Swarm Optimization

## Software Dependencies

To run this code, you'll need the following:

- Python 3.9+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook/Lab

You can install the required packages using pip:

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn jupyter
```

## Running the Code

The project is organized into Jupyter notebooks that should be run in sequence:

1. **2. Data Preparation.ipynb**: Prepares and preprocesses the exchange rate data
2. **3. Base Model.ipynb**: Implements the baseline MLP model
3. **4. MLP Univariate.ipynb**: Implements MLP for univariate time series prediction
4. **9. PSO MLP Univariate.ipynb**: Implements PSO-optimized MLP for univariate prediction
5. **8. Comparison.ipynb**: Compares the performance of different models

To run the notebooks:
1. Start Jupyter: `jupyter notebook` or `jupyter lab`
2. Open the notebooks in sequence and run all cells

## Interpreting the Outputs

### Model Performance Metrics

The models are evaluated using the following metrics:
- Root Mean Square Error (RMSE): Lower values indicate better performance
- Mean Absolute Error (MAE): Lower values indicate better performance
- Directional Accuracy: Higher values indicate better performance

### Visualization

The notebooks generate various plots:
- Actual vs. Predicted values
- Error distributions
- PSO optimization progress
- Comparative performance charts

These plots are saved in the `PLOTS/` directory.

### PSO Optimization

The PSO optimization process tunes the following hyperparameters:
- Number of layers
- Number of neurons per layer
- Activation functions
- Learning rate
- Dropout rate

The best hyperparameter configuration is reported in the PSO notebook along with the optimization progress.

## Notes

- The dataset contains exchange rate data from 2001 onwards
- Models use a 50-day lag for prediction
- The test set consists of the last 50 observations in the dataset 