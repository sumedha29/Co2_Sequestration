# Co2 Sequestration Project

This project contains code for the CO2 Sequestration project. The code contains data-driven and physics-based MLP and LSTM models. The model is made available for the purpose of testing, and does not contain code for K-fold cross validation to allow faster training.

Input features:
* features1: time, position, permeability, porosity and injection rate
* features2: time, permeability and injection rate values from the .out files (Ks and Rs)
* features3: 3x3 matrix of permeability values (K in Equation (1) and (2) from the paper)

The input features to the neural networks are time, position, permeability, porosity and injection rate. These are denoted by features1 and features2 in the code. features1 are given as input to the single-output data driven MLP and LSTM models. features1 and features2 are given as input to the joint data driven MLP and LSTM models. In order to incorporate physics to the model, we have built features3 to represent the permeability matrix used in the calculations of the physics equations. Therefore, features3 is passed as an additional set of input features to the physics based MLP and LSTM models. 

