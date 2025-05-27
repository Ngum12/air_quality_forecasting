Collecting workspace information# Beijing Air Quality Forecasting Project

## Project Overview

This project implements a deep learning approach to forecast PM2.5 air pollution levels in Beijing using historical meteorological and air quality data. Accurate prediction of air pollution levels is critical for public health planning, allowing authorities to issue timely warnings and implement preventive measures when dangerous pollution levels are anticipated.

The solution employs bidirectional LSTM neural networks to capture complex temporal patterns in air quality data, enhanced with extensive feature engineering and careful handling of the time series characteristics.

## Dataset Description

The dataset contains hourly measurements of multiple atmospheric variables in Beijing:

- **PM2.5**: Fine particulate matter concentration (target variable)
- **Meteorological measurements**: Temperature, pressure, dew point, etc.
- **Wind data**: Wind speed and direction categories
- **Temporal information**: Timestamps from which various time-based features are derived

Data analysis showed that PM2.5 values exhibit significant positive skewness (1.81) and high kurtosis (5.10), indicating a right-tailed distribution with outliers. The dataset contained approximately 6.3% missing values in the PM2.5 column.

## Technical Approach

### Data Preprocessing

- **Missing value handling**: Linear interpolation followed by forward/backward fill to ensure continuous time series
- **Outlier analysis**: Quantile analysis to identify extreme pollution events
- **Temporal alignment**: Consistent datetime indexing for proper time series handling
- **Seasonal decomposition**: Analysis of daily, weekly, and monthly patterns

### Feature Engineering

1. **Temporal features**:
   - Extraction of hour, day, month, weekday, and season
   - Cyclical encoding using sine/cosine transformations to preserve circular nature of time variables

2. **Contextual indicators**:
   - Weekend/weekday flags
   - Rush hour periods
   - Night hours
   - Seasonal indicators (winter flag)

3. **Lag features**:
   - Historical PM2.5 values at different time lags (1, 3, 6, 12, 24 hours)
   - Rolling statistics (mean, standard deviation) with windows of 6, 12, and 24 hours

### Model Architecture

The final model uses a sophisticated neural network architecture:

```
Input Layer
│
├─ Bidirectional LSTM (128 units, return sequences=True)
│   └─ Dropout (0.3) & Recurrent Dropout (0.2)
│
├─ Bidirectional LSTM (64 units)
│   └─ Dropout (0.3) & Recurrent Dropout (0.2)
│
├─ Dense Layer (32 units, ReLU activation)
│   └─ Dropout (0.2)
│
Output Layer (1 unit)
```

Key model characteristics:
- **Bidirectional processing**: Captures patterns from both past and future context
- **Multiple LSTM layers**: Hierarchical feature extraction
- **Regularization**: Strategic dropout to prevent overfitting
- **Adam optimizer**: With tuned learning rate of 5e-4
- **Adaptive learning rate**: Reduced by factor of 0.5 when validation performance plateaus

## Results and Evaluation

The model was evaluated using Root Mean Squared Error (RMSE), with training monitored through:

- Training and validation loss curves
- Learning rate adaptation visualization
- Distribution analysis of predictions
- Time series plots of forecasted values

The RMSE training curves show consistent improvement with effective convergence, suggesting the model successfully captures the underlying patterns in PM2.5 concentration.

## Installation and Usage

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn

### Running the Project
1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook air_quality_forecasting.ipynb`

### Making Predictions
The notebook includes code for generating predictions on new data and creating submission files in the expected format:

```python
# Prepare test data
X_test_scaled = scaler_x.transform(X_test)
X_test_scaled = np.expand_dims(X_test_scaled, axis=1)

# Generate predictions
predictions = model.predict(X_test_scaled)
predictions = scaler_y.inverse_transform(predictions)
```

## Future Improvements

Several avenues could further enhance the model's performance:

1. **Advanced architectures**: Testing Transformer models or hybrid CNN-LSTM approaches
2. **External data integration**: Incorporating traffic patterns, industrial activity logs, or holiday calendars
3. **Ensemble methods**: Combining multiple models for improved robustness
4. **Specialized handling of extreme events**: Targeted modeling of pollution spikes
5. **Hyperparameter optimization**: Systematic search using Bayesian optimization or genetic algorithms

## Authors

- Your Name

## License

This project is licensed under the MIT License - see the LICENSE file for details.
