# Code Explanation

Loading the diabetes dataset from scikit learn datasets

`diabetes = datasets.load_diabetes()`

Use only one feature from the dataset for the regression

`diabetes_X = diabetes.data[:, np.newaxis, 2]`

Split the data and targets into training and testing sets

```sh diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:] 
```

Create linear regression model (regr), train using training set (.fit method), and do the predictions using the test set

```sh
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
diabetes_y_pred = regr.predict(diabetes_X_test)
```