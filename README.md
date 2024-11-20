# House Price Prediction Model

This project builds a machine learning model to predict house prices using the Ames Housing dataset. The dataset includes detailed information about houses, such as their size, quality, neighborhood, and various other features. The goal is to create an accurate model that can estimate house prices based on these features.


## Dataset

The dataset used in this project is the **Ames Housing dataset**, which contains 82 features and 2,930 observations about residential properties in Ames, Iowa. The target variable is `SalePrice`, representing the sale price of the house in USD.

Key aspects of the dataset:
- **Target Variable:** `SalePrice`
- **Features:** Mixture of numerical and categorical variables, including house size, neighborhood, year built, and more.
- **Source:** [Ames Housing Dataset](http://jse.amstat.org/v19n3/decock.pdf)

---

## Features

### Preprocessing Steps:
1. **Numerical Features:** 
   - Missing values are imputed with the mean.
   - Features are standardized using `StandardScaler`.

2. **Categorical Features:**
   - Missing values are imputed with the most frequent value.
   - Features are encoded using one-hot encoding.

### Target Variable:
- `SalePrice`: The sale price of the house in USD.

---

## Model

The model is implemented using a **Random Forest Regressor**, which is an ensemble learning technique that uses multiple decision trees for improved accuracy. The pipeline includes:
1. Data Preprocessing (handling missing values and encoding).
2. Feature Scaling and Transformation.
3. Random Forest Regressor for predictions.

---

## Results

The model's performance metrics are as follows:
- **Mean Absolute Error (MAE):** \$15,712.35
- **Root Mean Squared Error (RMSE):** \$26,140.10
- **R² Score:** 0.915

This indicates that the model explains 91.5% of the variance in house prices and achieves reasonable prediction accuracy.

---

## Setup Instructions

### Prerequisites
- Python 3.8 or later
- Libraries: `pandas`, `numpy`, `scikit-learn`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd house-price-prediction
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Place the dataset (`AmesHousing.csv`) in the project folder.
5. Run the script:
   ```bash
   python house_price_prediction.py
   ```

---

## Future Work

- **Model Optimization:** Experiment with hyperparameter tuning for Random Forest.
- **Additional Models:** Test other algorithms like Gradient Boosting or Neural Networks.
- **Feature Engineering:** Derive new features or remove redundant ones for better accuracy.
- **Visualization:** Add exploratory data analysis (EDA) and visualizations for better insights.
