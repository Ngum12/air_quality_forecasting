# Beijing Air Quality Forecasting Project

## Project Overview

This project implements a deep learning approach to forecast PM2.5 air pollution levels in Beijing using historical meteorological and air quality data. Air pollution, particularly PM2.5 particulate matter, represents a critical global health concern that affects respiratory health and quality of life in urban areas. Accurate prediction of air pollution levels is essential for public health planning, allowing authorities to issue timely warnings and implement preventive measures when dangerous pollution levels are anticipated.

The solution employs bidirectional LSTM neural networks to capture complex temporal patterns in air quality data, enhanced with extensive feature engineering and careful handling of the time series characteristics. This approach was selected because of LSTM's proven effectiveness in modeling long-range dependencies in sequential data, making it particularly suitable for environmental time series with strong daily, weekly, and seasonal patterns.

## Dataset Description

The dataset contains hourly measurements of multiple atmospheric variables in Beijing from January 1st, 2010 to December 31st, 2014:

- **PM2.5**: Fine particulate matter concentration in μg/m³ (target variable)
- **Meteorological measurements**: Temperature (TEMP), pressure (PRES), dew point (DEWP)
- **Wind data**: Wind speed (Iws) and direction categories (cbwd)
- **Precipitation**: Is (cumulated hours of snow), Ir (cumulated hours of rain)
- **Temporal information**: Timestamps from which various time-based features are derived

### Dataset Statistics:
- **Total observations**: 43,824 hourly records (5 years)
- **Missing values**: Approximately 6.3% missing values in the PM2.5 column
- **Target distribution**: PM2.5 values exhibit significant positive skewness (1.81) and high kurtosis (5.10)
- **Percentile analysis**: 
  - 25th percentile: 22.0 μg/m³
  - Median: 72.0 μg/m³  
  - 75th percentile: 121.0 μg/m³
  - 99th percentile: 480.0 μg/m³ (indicating extreme pollution events)

## Technical Approach

### Data Exploration

Before modeling, we conducted extensive exploratory data analysis to understand the data characteristics:

1. **Time series visualization**: Identified clear daily, weekly, and seasonal patterns in PM2.5 levels
2. **Correlation analysis**: Found strong relationships between PM2.5 and weather variables (particularly temperature and wind speed)
3. **Hourly patterns**: Identified peak pollution periods typically occurring during morning (7-9 AM) and evening (5-7 PM) rush hours
4. **Seasonal trends**: Winter months showed significantly higher pollution levels due to increased heating demands and weather conditions
5. **Missing data patterns**: Visualized using missingno to identify any systematic patterns in missing values

These insights directly informed our preprocessing and feature engineering strategies.

### Data Preprocessing

- **Missing value handling**: Linear interpolation followed by forward/backward fill to ensure continuous time series. This approach was selected to preserve temporal patterns better than simple mean imputation.
- **Outlier analysis**: Quantile analysis to identify extreme pollution events (>480 μg/m³). These values were retained rather than capped as they represent legitimate pollution spikes.
- **Temporal alignment**: Consistent datetime indexing for proper time series handling, ensuring 1-hour intervals
- **Seasonal decomposition**: Applied additive decomposition to separate trend, seasonality, and residual components at daily (24h), weekly (168h), and monthly (720h) granularities, providing insights about underlying patterns

### Feature Engineering

1. **Temporal features**:
   - Extraction of hour, day, month, weekday, and season from timestamps
   - Cyclical encoding using sine/cosine transformations to preserve circular nature of time variables:
     ```python
     for col, period in [('hour', 24), ('month', 12), ('weekday', 7), ('day', 30)]:
         df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
         df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
     ```

2. **Contextual indicators**:
   - Weekend/weekday flags: `(df['weekday'] >= 5).astype(int)`
   - Rush hour periods: `df['hour'].between(7, 9) | df['hour'].between(17, 19)`
   - Night hours: `(df['hour'] >= 22) | (df['hour'] <= 5)`
   - Seasonal indicators: `df['is_winter'] = df['month'].isin([12, 1, 2])`

3. **Lag features**:
   - Historical PM2.5 values at different time lags (1, 3, 6, 12, 24 hours)
   - Rolling statistics with various windows:
     - Mean: `df['pm25_roll_mean_24'] = df['pm2.5'].rolling(24).mean()`
     - Standard deviation: `df['pm25_roll_std_24'] = df['pm2.5'].rolling(24).std()`
     - Exponentially weighted moving average: `df['pm25_ewma_24'] = df['pm2.5'].ewm(span=24).mean()`
     - Rolling quantiles (25%, 75%)

4. **Interaction features**:
   - Temperature-dewpoint interaction: `df['temp_dewp_interaction'] = df['TEMP'] * df['DEWP']`
   - Wind-temperature interaction: `df['wind_temp_interaction'] = df['Iws'] * df['TEMP']`

### Input Sequence Preparation

To leverage the temporal nature of the data, we prepared sequences for the LSTM model:

1. **Target definition**: Forecasting PM2.5 concentration 1-24 hours ahead
2. **Feature normalization**: StandardScaler applied to both features and target
3. **Sequence reshaping**: Converted to 3D format (samples, timesteps, features) required by LSTM

## Model Architecture

After extensive experimentation, the final model employs a sophisticated neural network architecture:

```
Input Layer (shape=(samples, timesteps, features))
│
├─ Bidirectional LSTM (128 units, return sequences=True)
│   └─ Dropout (0.3) & Recurrent Dropout (0.2)
│
├─ BatchNormalization
│
├─ Bidirectional LSTM (64 units)
│   └─ Dropout (0.3) & Recurrent Dropout (0.2)
│
├─ Dense Layer (32 units, ReLU activation)
│   └─ Dropout (0.2)
│
Output Layer (1 unit, linear activation)
```

Key model characteristics:
- **Bidirectional processing**: Captures patterns from both past and future context
- **Multiple LSTM layers**: Hierarchical feature extraction
- **BatchNormalization**: Improves training stability and addresses internal covariate shift
- **Regularization**: Strategic dropout to prevent overfitting
- **Gradient clipping**: Set to 1.0 to prevent exploding gradients, a common issue in RNNs
- **Adam optimizer**: With tuned learning rate of 5e-4
- **Adaptive learning rate**: Reduced by factor of 0.5 when validation performance plateaus

### Addressing RNN Challenges

- **Vanishing gradients**: Addressed through LSTM architecture that includes forget, input, and output gates to control information flow
- **Exploding gradients**: Mitigated with gradient clipping (clipnorm=1.0)
- **Overfitting**: Combated with dropout layers, recurrent dropout, and early stopping
- **Training instability**: Addressed with batch normalization and careful learning rate scheduling

## Experimental Results

We conducted extensive hyperparameter tuning and architecture exploration, summarized in the following experiment table:

| Experiment | Architecture | Learning Rate | Batch Size | Dropout | Regularization | Training RMSE | Validation RMSE |
|------------|--------------|--------------|------------|---------|----------------|---------------|-----------------|
| 1 | LSTM(64) | 1e-3 | 32 | 0.2 | None | 0.786 | 0.745 |
| 2 | LSTM(128) | 1e-3 | 32 | 0.2 | None | 0.732 | 0.703 |
| 3 | BiLSTM(64) | 1e-3 | 32 | 0.2 | None | 0.711 | 0.687 |
| 4 | BiLSTM(128) | 1e-3 | 32 | 0.3 | None | 0.684 | 0.674 |
| 5 | BiLSTM(128) | 5e-4 | 32 | 0.3 | L2(1e-5) | 0.641 | 0.595 |
| 6 | BiLSTM(128) | 5e-4 | 64 | 0.3 | L2(1e-5) | 0.648 | 0.591 |
| 7 | BiLSTM(128)+BiLSTM(64) | 5e-4 | 32 | 0.3 | L2(1e-5) | 0.589 | 0.554 |
| 8 | BiLSTM(128)+BiLSTM(64) | 5e-4 | 32 | 0.3 | L2(1e-5)+BN | 0.562 | 0.483 |
| 9 | BiLSTM(128)+BiLSTM(64) | 2.5e-4 | 32 | 0.3 | L2(1e-5)+BN | 0.523 | 0.447 |
| 10 | BiLSTM(128)+BiLSTM(64)+Dense(32) | 5e-4 | 32 | 0.3 | L2(1e-5)+BN | 0.412 | 0.386 |
| 11 | BiLSTM(192)+BiLSTM(96)+Dense(32) | 5e-4 | 32 | 0.3 | L2(1e-5)+BN | 0.397 | 0.376 |
| 12 | BiLSTM(192)+BiLSTM(96)+LSTM(48)+Dense(32) | 5e-4 | 32 | 0.3 | L2(1e-5)+BN | 0.389 | 0.363 |
| 13 | BiLSTM(128)+BiLSTM(64)+Dense(32) | 5e-4 | 32 | 0.2 | L2(1e-6)+BN | 0.377 | 0.359 |
| 14 | BiLSTM(128)+BiLSTM(64)+Dense(32) | 5e-4 | 64 | 0.2 | L2(1e-6)+BN | 0.368 | 0.342 |
| 15 | BiLSTM(128)+BiLSTM(64)+Dense(32) | Adaptive* | 64 | 0.2 | L2(1e-6)+BN | 0.362 | 0.282 |

*Adaptive learning rate: Initial rate of 5e-4 with ReduceLROnPlateau (factor=0.5, patience=7)
*BN = BatchNormalization layers

### RMSE Definition and Analysis

Root Mean Squared Error (RMSE) was used as the primary evaluation metric:

RMSE = √(1/n ∑(yᵢ - ŷᵢ)²)

Where:
- n is the number of observations
- yᵢ is the actual PM2.5 value
- ŷᵢ is the predicted PM2.5 value

Lower RMSE values indicate better model performance. Our best model achieved a validation RMSE of 0.282 (after denormalization: approximately 29.4 μg/m³).

### Learning Insights from Experiments

1. **Architecture complexity**: Adding bidirectional layers consistently improved performance by capturing patterns in both directions
2. **Regularization importance**: Batch normalization had the most significant impact on reducing validation error
3. **Hyperparameter sensitivity**: Learning rate adjustments were crucial, with adaptive schedules outperforming fixed rates
4. **Overfitting control**: The combination of dropout (0.2), L2 regularization (1e-6), and early stopping provided optimal balance

## Results and Evaluation

The model was evaluated using various techniques:

- **Training and validation loss curves**: Showed consistent improvement with effective convergence
- **Learning rate adaptation**: Visualization confirmed automatic reduction at appropriate plateaus
- **Distribution analysis**: Predicted PM2.5 values followed expected patterns including diurnal variations
- **Time series plots**: Demonstrated accurate tracking of pollution trends and seasonality

The final model successfully captured:
1. Daily pollution patterns with morning and evening peaks
2. Weekly variations with lower weekend pollution levels
3. Seasonal trends with winter pollution spikes
4. Responsiveness to weather conditions, particularly wind speed changes

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
The notebook includes code for generating predictions on new data:

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

1. **Advanced architectures**: Testing Transformer models or hybrid CNN-LSTM approaches for capturing both local and global patterns
2. **External data integration**: Incorporating traffic patterns, industrial activity logs, or holiday calendars
3. **Ensemble methods**: Combining multiple models for improved robustness and performance
4. **Specialized handling of extreme events**: Targeted modeling of pollution spikes through quantile regression or specialized loss functions
5. **Hyperparameter optimization**: Systematic search using Bayesian optimization or genetic algorithms

## Conclusion

This project successfully demonstrates the application of bidirectional LSTM networks to forecast air pollution levels with high accuracy. By carefully engineering temporal features and optimizing the neural network architecture, we achieved a validation RMSE of 0.282 (normalized scale).

Key findings include the importance of bidirectional processing for environmental time series, the critical role of feature engineering in capturing pollution patterns, and the effectiveness of regularization techniques in preventing overfitting. The methodology can be extended to other cities and environmental forecasting problems with appropriate adaptations.

## References

[1] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, no. 7553, pp. 436-444, 2015.

[2] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735-1780, 1997.

[3] A. Graves and J. Schmidhuber, "Framewise phoneme classification with bidirectional LSTM and other neural network architectures," *Neural Networks*, vol. 18, no. 5-6, pp. 602-610, 2005.

[4] Z. Zhao, W. Chen, X. Wu, P. C. Y. Chen, and J. Liu, "LSTM network: a deep learning approach for short-term traffic forecast," *IET Intelligent Transport Systems*, vol. 11, no. 2, pp. 68-75, 2017.

[5] World Health Organization, "Air quality guidelines: global update 2005," WHO Regional Office for Europe, Copenhagen, 2006.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
