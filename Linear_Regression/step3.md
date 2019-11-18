# Code Explanation

`diabetes = datasets.load_diabetes()`

Loading the scikit learn dataset

`diabetes_X = diabetes.data[:, np.newaxis, 2]`

Use only one feature for the regression
```sh diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:] 
```
Split the data and targets into training and testing sets

```sh
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
diabetes_y_pred = regr.predict(diabetes_X_test)
```

Create linear regression model, train using training set, and do the predictions using the test set