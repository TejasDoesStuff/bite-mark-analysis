"""
Bite Mark Feature Interpreter (BMFI)

Inputs:
Alterior Teeth Width (mm),
Alterior Teeth Depth (mm),
Bite Force(N)

Outputs:
Weight (kg),
Height (cm),
Jaw Size (cm),
Race,
Gender
"""

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)

# reading the files
df = pd.read_csv('bite_mark_data.csv')

X = df[['Left Tooth Depth (mm)', 'Right Tooth Depth (mm)', 'Left Tooth Width (mm)', 'Right Tooth Width (mm)', 'Bite Force (N)']]
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly = sm.add_constant(X_poly)

y_vars = ['Weight (kg)', 'Height (cm)', 'Jaw Size (cm)', 'Race', 'Gender']
results = {}

# creating a regression model for each y variable
for var in y_vars:
    y = df[var]
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
    model = sm.OLS(y_train, X_train).fit()
    results[var] = {'model': model, 'X_test': X_test, 'y_test': y_test}

all_y_test = []
all_predictions = []

# graphing + stat output
plt.figure(figsize=(15, 10))

for i, (var, data) in enumerate(results.items()):
    model = data['model']
    X_test = data['X_test']
    y_test = data['y_test']

    predictions = model.predict(X_test)

    all_y_test.extend(y_test)
    all_predictions.extend(predictions)

    plt.subplot(2, 3, i + 1)

    plt.scatter(y_test, predictions, color='blue', alpha=0.5, s=30)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted: {var}')
    plt.grid()

    sorted_indices = np.argsort(y_test)
    sorted_y_test = y_test.iloc[sorted_indices]
    sorted_predictions = predictions[sorted_indices]

    lowess = sm.nonparametric.lowess(sorted_predictions, sorted_y_test, frac=0.3)
    plt.plot(lowess[:, 0], lowess[:, 1], color='orange', linewidth=2, label='LOWESS Smoothing')

    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction')

    r_squared = model.rsquared
    adjusted_r_squared = model.rsquared_adj
    plt.text(0.05, 0.95, f'R²: {r_squared:.4f}\nAdj. R²: {adjusted_r_squared:.4f}',
             transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.5))

    if var == 'Race':
        plt.xlim(0, 3)
        plt.ylim(0, 3)

plt.tight_layout()
plt.show()
