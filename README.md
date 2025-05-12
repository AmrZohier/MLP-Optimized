# Exchange Rate Prediction with PSO-Optimized Neural Networks

This project implements and compares various neural network architectures for predicting exchange rates, with a focus on GBP/USD currency pair. The models are optimized using Particle Swarm Optimization (PSO) to find optimal hyperparameters.

## Project Structure

- **Data Preparation**: Processing and merging financial data from multiple sources
- **Model Implementation**: MLP and CNN architectures in both univariate and multivariate configurations
- **PSO Optimization**: Custom implementation for neural network hyperparameter tuning

## Notebooks

1. **Data Preparation**: Data collection and preprocessing
2. **MLP Univariate**: Basic MLP model with univariate input
3. **CNN Univariate**: Basic CNN model with univariate input
4. **PSO MLP Univariate**: MLP model optimized with PSO
5. **PSO CNN Univariate**: CNN model optimized with PSO

## Data Sources

- **Normalized GDP** (monthly): Federal Reserve Economic Data (FRED)
- **Libor Rates** (daily): Federal Reserve Economic Data (FRED)
- **Current Account to GDP** (quarterly): OECD Statistics
- **Forex** (daily): Federal Reserve

## Key Features

- Custom implementation of Particle Swarm Optimization for neural network hyperparameter tuning
- Comparison between traditional and PSO-optimized models
- Implementation of the Swish activation function
- Time series forecasting with lagged sequences

## Requirements

- Python 3.11+
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Results

The PSO-optimized models demonstrate significant improvements over the baseline models:
- Original MLP Univariate RMSE: 0.01295
- PSO-Optimized MLP Univariate: 0.00708 45.33% improvement
- Original CNN Univariate RMSE: 0.03028
- PSO-Optimized CNN Univariate: 0.00654 78.40% improvement

## Usage

1. Run the data preparation notebook to generate the merged dataset
2. Execute the model notebooks to train and evaluate different architectures
3. Use the PSO optimizer notebooks to find optimal hyperparameters and train improved models