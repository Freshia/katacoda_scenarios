# Let's take a look at our dataset

Open the file `myprogram/data_exploration.py` and paste the following

<pre class="file" data-filename="myprogram/data_exploration.py" data-target="replace">
import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn import datasets,preprocessing,cross_validation,neighbors,model_selection
breasts = datasets.load_breast_cancer()
print("KEYS")
print(breasts.keys())
print("DATA")
print(breasts.data)
print("THE TARGET\n")
print(breasts.target)
print("DESCRIPTION")
print(breasts.DESCR)
print("TARGET NAMES")
print(breasts.target_names)
print(breasts.feature_names)
</pre>

The above code gives details about the breast cancer dataset in scikit learn

To run the code, run the following in the terminal:

`python myprogram/data_exploration.py`{{execute}}
