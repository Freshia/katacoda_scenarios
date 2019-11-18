# Testing our model with sample data

Open the file  `myprogram/k_neighbours.py` and paste the following lines below the code in the file

```sh

example_measures= np.array([1.908e+01,2.571e+01,7.563e+01,5.900e+02,2.075e-01, 1.570e-01,4.128e-02, 3.110e-02,1.157e-01,
5.261e-02,2.052e-01,8.477e-01,2.183e+00,1.567e+01,
4.097e-03,2.498e-02,1.198e-02,6.490e-03,1.178e-02,2.425e-03,1.450e+01,2.049e+01,7.609e+01,5.305e+02,
2.512e-01,3.876e-01,1.390e-01,7.683e-02,3.364e-01,2.183e-02])


example_measures=example_measures.reshape(1,-1)
prediction=clf.predict(example_measures)
print(prediction)

```

The above code tests the K nearest neighbours model with sample data
The model the classifies the data and prints the prediction


To run the code, run the following in the terminal:

`python myprogram/k_neighbours.py`{{execute}}