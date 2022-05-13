from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
    RANSAC Algorithm
    
1.Select a random number of examples to be inliers and train the model.

2.Test all other data points against the trained model

3.Out of all the data points tested in step 2, select the points as inliers which fall within a user-given tolerance. 
  In scikit-learn, median absolute deviation (MAD) is used for selecting the new points as inliers.

4.Retrain the model with all inliers data

5.Estimate the error of the retrained model versus the inliers.

6.Follow step 1 to step 5

7.Terminate the algorithm execution if the model performance meets a certain user-defined threshold 
  or if a fixed number of iterations were reached
"""


#
# Load the Boston Housing dataset for training
#
bhd = datasets.load_boston()
df = pd.DataFrame(bhd.data)
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
               'TAX', 'PTRATIO', 'B', 'LSTAT']
df['MEDV'] = bhd.target
#
# Select Avg. No of rooms per dwelling as feature
# and fit the model
#
X = df['RM'].to_numpy().reshape(-1, 1)
y = df['MEDV'].to_numpy().reshape(-1, 1)
#
# Create an instance of RANSACRegressor
#
ransac = RANSACRegressor(base_estimator=LinearRegression(),
                         min_samples=50, max_trials=100,
                         loss='absolute_loss', random_state=42,
                         residual_threshold=10)
#
# Fit the model
#
ransac.fit(X, y)

#
# Get the Inlier mask; Create outlier mask
#
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
#
# Create scatter plot for inlier datset
#
plt.figure(figsize=(8, 8))
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white',
            marker='o', label='Inliers')
#
# Create scatter plot for outlier datset
#
plt.scatter(X[outlier_mask], y[outlier_mask],
             c='limegreen', edgecolor='white',
             marker='s', label='Outliers')
#
# Draw the best fit line
#
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Average number of rooms [RM]', fontsize=15)
plt.ylabel('Price in $1000s [MEDV]', fontsize=15)
plt.legend(loc='upper left', fontsize=12)
plt.show()