# Air Quality Forecasting: PM2.5 Prediction Project

## Project Overview

This repository contains a time series forecasting solution for predicting PM2.5 air pollution levels. The model uses historical air quality data along with weather conditions to forecast future PM2.5 concentrations, which is critical for public health monitoring and environmental management.

## Table of Contents

1. Dataset Description
2. Technical Approach
3. Model Architecture
4. Feature Engineering
5. Results and Performance
6. Installation and Usage
7. Future Improvements

## Dataset Description

The dataset consists of hourly measurements from July 2013 to December 2014, including:

- **PM2.5 concentrations** (target variable)
- **Weather conditions** (temperature, pressure, wind speed, etc.)
- **Time-based features** (hour, day, month)

The data exhibits strong temporal patterns with both daily and seasonal variations, making it suitable for time series modeling approaches.

## Technical Approach

Our solution employs a deep learning approach using Long Short-Term Memory (LSTM) networks, which are particularly effective for time series forecasting due to their ability to capture long-term dependencies in sequential data.

Key steps in our approach:
1. Extensive feature engineering to capture temporal patterns
2. Data preprocessing including normalization and sequence creation
3. Training multiple LSTM architectures with different hyperparameters
4. Model evaluation using RMSE (Root Mean Square Error)

## Model Architecture

The best-performing model uses a sequential architecture with:

```python
Sequential([
    LSTM(128, activation='tanh', return_sequences=True, input_shape=(sequence_length, n_features)),
    LSTM(64, activation='tanh'),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
```

This architecture was chosen after extensive experimentation with different configurations of layers, units, and regularization techniques. The model uses the Adam optimizer with a learning rate of 0.001.

## Feature Engineering

Feature engineering was critical to achieving good performance:

1. **Lag Features**: Past values of PM2.5 at intervals (1, 3, 6, 12, 24 hours) provide important context
2. **Rolling Statistics**: Moving averages and standard deviations over different windows (6, 12, 24 hours)
3. **Cyclical Time Features**: Hour of day and month encoded using sine/cosine transformations
4. **Sequential Approach**: Using 24-hour sequences to predict the next hour's PM2.5 level

## Results and Performance

Our model achieved an RMSE of ~4470, which indicates strong predictive performance. The model shows particular strength in:

- Capturing daily patterns of pollution levels
- Adapting to changing weather conditions
- Providing accurate next-hour predictions

Key insights:
- Lag features proved to be the most important predictors
- A sequence length of 24 hours provided the best balance between context and model complexity
- Log-transforming the target variable helped handle the skewed distribution of PM2.5 values

## Installation and Usage

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Pandas, NumPy, Matplotlib
- Scikit-learn

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/air-quality-forecasting.git
cd air-quality-forecasting

# Install dependencies
pip install -r requirements.txt
```

### Running the Code
1. Place the training and test datasets in the `data/` folder
2. Run the feature engineering script:
   ```bash
   python feature_engineering.py
   ```
3. Train the model:
   ```bash
   python train_model.py
   ```
4. Generate predictions:
   ```bash
   python predict.py
   ```

## Future Improvements

Several avenues could further enhance model performance:
1. Incorporating additional external data (traffic patterns, industrial activity)
2. Ensemble techniques combining multiple model predictions
3. More sophisticated handling of extreme events and outliers
4. Advanced time series techniques like attention mechanisms

---

## Contributors
- [Your Name]

## License
This project is licensed under the MIT License - see the LICENSE file for details.
