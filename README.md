# -Student-Score-Prediction
Predict student exam scores using regression models. Compares Linear vs Polynomial Regression with feature engineering. Best results: Linear Regression with 10 features (R² = 0.614).

## Dataset

* **Source:** Student Performance Factors (Kaggle)
* **Description:** Academic and behavioral attributes of students used to predict exam scores.


## Methodology

1. **Data Cleaning & EDA** – handled missing values and visualized feature relationships.
2. **Feature Engineering** – experimented with different combinations (e.g., sleep, participation).
3. **Model Training** – applied Linear Regression and Polynomial Regression.
4. **Evaluation** – measured performance using R² score and error metrics.


## Results

* **Polynomial Regression** consistently failed to improve performance.
* **Linear Regression** improved significantly with more relevant features.
* **Best Setup:** Linear Regression with 10 features → **R² = 0.614** (highest accuracy, lowest error).

## Tools & Libraries

* Python
* Pandas – data handling
* Matplotlib – visualization
* Scikit-learn – regression & evaluation

##  Conclusion

Linear Regression with 10 carefully chosen features outperformed Polynomial Regression and achieved the best prediction results. It provides the most reliable and generalizable model for student score prediction in this dataset.

