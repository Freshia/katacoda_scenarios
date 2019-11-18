# Creating our K Nearest Neighbours Model

Open the file `myprogram/k_neighbours.py` and paste the following

<pre class="file" data-filename="myprogram/linear_regression.py" data-target="replace">

import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn import datasets,preprocessing,cross_validation,neighbors,model_selection
breasts = datasets.load_breast_cancer()

x=np.array(breasts.data)
y=np.array(breasts.target)

x_train,x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

</pre>

The above code loads the breast cancer dataset from scikit learn, creates a k nearest neighbours model, tests it and prints the accuracy

To run the code, run the following in the terminal:

`python myprogram/k_neighbours.py`{{execute}}