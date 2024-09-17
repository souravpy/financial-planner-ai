# Datasets for Machine Learning Models

## What are datasets

Datasets are simply a collection of feature names and their respective values for each index often stored in a csv format though there are many more used for various optimization beneifts, but they contain some sort of inputs and outputs.

```csv
Front Rideheight,Rear Rideheight,Raw Downforce Mean,Raw Drag Mean
6.442514877,4.089697725,101.3770801,49.75775049
5.66649446,4.811475098,114.7736165,51.80269056
5.278356263,5.172482827,120.3391432,52.07643129
4.825570796,5.593619047,125.4008859,52.46839927
4.372703189,6.014831666,123.9335348,52.04939417
```

For csv files, the first row in the file is called the header and contains the 'feature' names which define what the values in each column are referring to. In this example, the first column is labeled as "Front Rideheight", so we know every value in this column is a value of "Front Rideheight".

## Types of data

For most entry level applications, there are two types of data: numerical, and categorical. Broadly, numerical refers to anything that can be explicitly expressed as a number, and categorical often refers to strings though any non-number could fall under this.

Generally, any categorical features will need to be converted into a numerical form for the model to train off of.

## How are datasets used

Machine learning models on the entry level take in "training data" to "train" the model and "test data" to "test" the model's performance. Here training data encompasses a large set of inputs and known outputs from each row of inputs while test data only includes a smaller set of inputs and outputs.

### Training Data

Training data is the primary mechanism for how the model learns and improves. It is prefered to have this set of data be as large as possible while still contain useful inputs and cost efficient.

Training data is broken down into two components: input columns, and target columns. We want to keep the target column seperate when we go to train the model. The output is used to inform the model how close the current prediction is to this known output. This allows the model to tune its prediction to approach a minimal error.

### Test Data

Test data operates very similarly to training data with the only exception being size of the set.

Test data is also broken down into input columns and target column, but the target is only used for evaluation purposes and the model never sees the target column for any sort of training.

### Validation Data

Validation data is often a subset of the training data which is split from that set to form this one. This set is very beneficial to analyzing if the model is 'overfitting' the training data or not, essentially acting as a evaluation before the test set.
