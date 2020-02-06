# Forest-Type-Classification
In this assignment we are tasked to devise a predictive model that can allow the user to determine the correct forest type given several criteria about the forest. There are seven forest type our model has to make a prediction on and there are 55 attributes for each of the traning variables that indicates presence and absence of certain properties in the forest. We are provided with a data set that is split into traning data and testing data set of size 14,528 rows by 55 columns and 116,205 rows by 54 columns respectively. The traning data has class label columns that represent the underlying truth of the forest type of each variable. Using this data, we are tasked with to devise a classification model using Decision tree, Naïve Bayes, and Neural network that can best predict the forest type for our testing data. 
#### Bays classifier implementation
In order to devise a model that can predict the forest type given a traning data using naïve bays theorem. I have used Sklearn’ s already made naïve bays module that implement naive bays algorithm. Since our data is a mixture of binary variables and continuous data, and most of the variables are binary data, I tried first to use the Bernoulli naïve bay algorithm. Using the sklearn Bernoulli naïve bay module which is designed for binary and Boolean features, I fitted the half of the traning data and since I know the underlying truth for the traning data I was able to test the accuracy of the model on the other half of the traning data. The result was not satisfactory, the accuracy I was able to get was a 63%. Next, I tried to separate the continues data into its own group and the binary data into its own group and fitted the continues data using sklearn’ s gaussian classifier and the binary data using the same Bernoulli classifier. Once I fitted on both classifiers, I multiplied the probability output of each classifier and I was able to get 67% accuracy result in naïve bay’s model on the test data.
#### Decision trees model
For the decision tree model, I have used sklearn decision tree classifier module that implements the decision tree algorithm. In this model I was able to get an accuracy of 75 %. I have used Gini criteria to measure the quality of the data. 
#### Neural network model 
To build a neural network model that predicts the type of the forest I implemented the model using MLP classifier module of the sklearn library. In order to normalize and standardize the data I have used sklearn’ s standard scaler module which will make the mean and standard deviation of every columns to 0 and 1 respectively. I am able to get an accuracy of 75%. With a hidden layer size of (74,74,74) and a maximum iteration rate of 600.
Overall, I have tested neural network, bays theorem-based classifier and decision tree based classifier on the data set provided and the best model that best classify the forest type in my implementation is the neural network model. 