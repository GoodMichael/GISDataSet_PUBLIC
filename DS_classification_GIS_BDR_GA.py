#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import the library
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from deap import creator, base, tools, algorithms
import numpy as np
import random
import numpy  
from scipy import interpolate
import matplotlib.pyplot as plt

#set the random seed
random.seed(123)

"""
Step 1: Initialize data by pre-processing input
"""

#load data and generates the dataframe data. 
data = pd.read_table('gis-data.txt',sep=' ')

#input data normalization:

#AS aspect 0-90
data['AS'].replace(0,0,inplace=True)
data['AS'].replace(10,1/9,inplace=True)
data['AS'].replace(20,2/9,inplace=True)
data['AS'].replace(30,3/9,inplace=True)
data['AS'].replace(40,4/9,inplace=True)
data['AS'].replace(50,5/9,inplace=True)
data['AS'].replace(60,6/9,inplace=True)
data['AS'].replace(70,7/9,inplace=True)
data['AS'].replace(80,8/9,inplace=True)
data['AS'].replace(90,1,inplace=True)

#TP topological position 32 48 64 80 96
data['TP'].replace(16,0,inplace=True)
data['TP'].replace(32,0.2,inplace=True)
data['TP'].replace(48,0.4,inplace=True)
data['TP'].replace(64,0.6,inplace=True)
data['TP'].replace(80,0.8,inplace=True)
data['TP'].replace(96,1,inplace=True)

#AL altitude
AVE = data['AL'].mean()
MEAN = data['AL'].std()
data['AL']=data['AL'].apply(lambda x:(x-AVE)/MEAN)

#this column has been tested with two normalization forms related to paper.
#CA cos of aspect
AVE = data['CA'].mean()
MEAN = data['CA'].std()
data['CA']=data['CA'].apply(lambda x:(x-AVE)/MEAN)

#MAX = max(data['CA'])
#MIN = min(data['CA'])
#data['CA']= data['CA'].apply(lambda x:(x-MIN)/(MAX-MIN))

#this column has also been tested as inconsistent data column

#TE temporature 30,60,90
#AVE = data['TE'].mean()
#MEAN = data['TE'].std()
#data['TE']=data['TE'].apply(lambda x:(x-AVE)/MEAN)

data['TE'].replace(0,0,inplace=True)
data['TE'].replace(30,1/3,inplace=True)
data['TE'].replace(60,2/3,inplace=True)
data['TE'].replace(90,1,inplace=True)

#testing purposes 
#SL slope
#AVE = data['SL'].mean()
#MEAN = data['SL'].std()
#data['SL']=data['SL'].apply(lambda x:(x-AVE)/MEAN)

data['SL'].replace(10,0,inplace=True)
data['SL'].replace(20,1/7,inplace=True)
data['SL'].replace(30,2/7,inplace=True)
data['SL'].replace(40,3/7,inplace=True)
data['SL'].replace(50,4/7,inplace=True)
data['SL'].replace(60,5/7,inplace=True)
data['SL'].replace(70,6/7,inplace=True)
data['SL'].replace(80,7,inplace=True)

#RA rainfall
MAX = max(data['RA'])
MIN = min(data['RA'])
data['RA']= data['RA'].apply(lambda x:(x-MIN)/(MAX-MIN))

#AVE = data['RA'].mean()
#MEAN = data['RA'].std()
#data['RA']=data['RA'].apply(lambda x:(x-AVE)/MEAN)

#SA sin of aspect
MAX = max(data['SA'])
MIN = min(data['SA'])
data['SA']= data['SA'].apply(lambda x:(x-MIN)/(MAX-MIN))

#DS 
data['DS'].replace(10,0,inplace=True)
data['DS'].replace(90,1,inplace=True)

#try shuffle data for training randomly - success
data=data.sample(frac=1).reset_index(drop=True)

#randomly split data into training set and testing set -success 
msk = np.random.rand(len(data))<0.8
train_data = data[msk]
test_data  = data[~msk]

#test the correlation with four types of forests 
#here is the innitial part should be comment for checking the accuracy
#Specifically, dropping the number of these four columns will result in differnt results and generations.

data=data.drop(['RF'],axis=1)
data=data.drop(['SC'],axis=1)
data=data.drop(['WD'],axis=1)
data=data.drop(['WS'],axis=1)

"""
Step 2: send the pre-processing dataset into the model
"""
# Encode the classification labels to numbers
# Get classes and one hot encoded feature vectors as for five-class classification
le = LabelEncoder()
le.fit(data['DS'])
allClasses = le.transform(data['DS'])
allFeatures = data.drop(['DS'], axis=1)

# Form training, test, and validation sets
X_trainAndTest, X_validation, y_trainAndTest, y_validation = train_test_split(allFeatures, allClasses, test_size=0.20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_trainAndTest, y_trainAndTest, test_size=0.20, random_state=42)

# Feature subset fitness function
def getFitness(individual, X_train, X_test, y_train, y_test):

	# Parse our feature columns that we don't use
	# Apply one hot encoding to the features
	cols = [index for index in range(len(individual)) if individual[index] == 0]
	X_trainParsed = X_train.drop(X_train.columns[cols], axis=1)
	X_trainOhFeatures = pd.get_dummies(X_trainParsed)
	X_testParsed = X_test.drop(X_test.columns[cols], axis=1)
	X_testOhFeatures = pd.get_dummies(X_testParsed)

	# Remove any columns that aren't in both the training and test sets
	sharedFeatures = set(X_trainOhFeatures.columns) & set(X_testOhFeatures.columns)
	removeFromTrain = set(X_trainOhFeatures.columns) - sharedFeatures
	removeFromTest = set(X_testOhFeatures.columns) - sharedFeatures
	X_trainOhFeatures = X_trainOhFeatures.drop(list(removeFromTrain), axis=1)
	X_testOhFeatures = X_testOhFeatures.drop(list(removeFromTest), axis=1)

	# Apply logistic regression on the data, and calculate accuracy
	clf = LogisticRegression()
	clf.fit(X_trainOhFeatures, y_train)
	predictions = clf.predict(X_testOhFeatures)
	accuracy = accuracy_score(y_test, predictions)

	# Return calculated accuracy as fitness
	return (accuracy,)

#========DEAP GLOBAL VARIABLES (viewable by SCOOP)========

# Create Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(data.columns) - 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Continue filling toolbox...
toolbox.register("evaluate", getFitness, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

#========

def getHof():

	# Initialize variables to use eaSimple
	numPop = 10
	numGen = 10
	pop = toolbox.population(n=numPop)
	hof = tools.HallOfFame(numPop * numGen)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", numpy.mean)
	stats.register("std", numpy.std)
	stats.register("min", numpy.min)
	stats.register("max", numpy.max)

	# Launch genetic algorithm
	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=numGen, stats=stats, halloffame=hof, verbose=True)

	# Return the hall of fame
	return hof

def getMetrics(hof):

	# Get list of percentiles in the hall of fame
	percentileList = [i / (len(hof) - 1) for i in range(len(hof))]
	
	# Gather fitness data from each percentile
	testAccuracyList = []
	validationAccuracyList = []
	individualList = []
	for individual in hof:
		testAccuracy = individual.fitness.values
		validationAccuracy = getFitness(individual, X_trainAndTest, X_validation, y_trainAndTest, y_validation)
		testAccuracyList.append(testAccuracy[0])
		validationAccuracyList.append(validationAccuracy[0])
		individualList.append(individual)
	testAccuracyList.reverse()
	validationAccuracyList.reverse()
	return testAccuracyList, validationAccuracyList, individualList, percentileList


if __name__ == '__main__':

	'''
	First, we will apply logistic regression using all the features to acquire a baseline accuracy.
	'''
	individual = [1 for i in range(len(allFeatures.columns))]
	testAccuracy = getFitness(individual, X_train, X_test, y_train, y_test)
	validationAccuracy = getFitness(individual, X_trainAndTest, X_validation, y_trainAndTest, y_validation)
	print('\nTest accuracy with all features: \t' + str(testAccuracy[0]))
	print('Validation accuracy with all features: \t' + str(validationAccuracy[0]) + '\n')

	'''
	Now, we will apply a genetic algorithm to choose a subset of features that gives a better accuracy than the baseline.
	'''
	hof = getHof()
	testAccuracyList, validationAccuracyList, individualList, percentileList = getMetrics(hof)

	# Get a list of subsets that performed best on validation data
	maxValAccSubsetIndicies = [index for index in range(len(validationAccuracyList)) if validationAccuracyList[index] == max(validationAccuracyList)]
	maxValIndividuals = [individualList[index] for index in maxValAccSubsetIndicies]
	maxValSubsets = [[list(allFeatures)[index] for index in range(len(individual)) if individual[index] == 1] for individual in maxValIndividuals]

	print('\n---Optimal Feature Subset(s)---\n')
	for index in range(len(maxValAccSubsetIndicies)):
		print('Percentile: \t\t\t' + str(percentileList[maxValAccSubsetIndicies[index]]))
		print('Validation Accuracy: \t\t' + str(validationAccuracyList[maxValAccSubsetIndicies[index]]))
		print('Individual: \t' + str(maxValIndividuals[index]))
		print('Number Features In Subset: \t' + str(len(maxValSubsets[index])))
		print('Feature Subset: ' + str(maxValSubsets[index]))

	'''
	Now, we plot the test and validation classification accuracy to see how these numbers change as we move from our worst feature subsets to the 
	best feature subsets found by the genetic algorithm.
	'''
	# Calculate best fit line for validation classification accuracy (non-linear)
	tck = interpolate.splrep(percentileList, validationAccuracyList, s=5.0)
	ynew = interpolate.splev(percentileList, tck)

	e = plt.figure(1)
	plt.plot(percentileList, validationAccuracyList, marker='o', color='r')
	plt.plot(percentileList, ynew, color='b')
	plt.title('Validation Set Classification Accuracy vs. \n Continuum with Cubic-Spline Interpolation')
	plt.xlabel('Population Ordered By Increasing Test Set Accuracy')
	plt.ylabel('Validation Set Accuracy')
	e.show()

	f = plt.figure(2)
	plt.scatter(percentileList, validationAccuracyList)
	plt.title('Validation Set Classification Accuracy vs. Continuum')
	plt.xlabel('Population Ordered By Increasing Test Set Accuracy')
	plt.ylabel('Validation Set Accuracy')
	f.show()

	g = plt.figure(3)
	plt.scatter(percentileList, testAccuracyList)
	plt.title('Test Set Classification Accuracy vs. Continuum')
	plt.xlabel('Population Ordered By Increasing Test Set Accuracy')
	plt.ylabel('Test Set Accuracy')
	g.show()

    























# In[2]:


# import libraries
from torch.autograd import Variable
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np 
import random
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

#set random seed to 1 to ensure result is generated with epochs.

#set the features of the in puts-success should be 16 inputs neurons
n_features = train_data.shape[1]-6

#split training data into input and target output 
#train input varies from AS to T7 -success
train_input = train_data.iloc[:,1:n_features+1]


#train_target vary from SC, DS, WD, WS, RF -success
#train_target= train_data.iloc[:,n_features+1:n_features+6]
#incase I just use the DS column
train_target= train_data.iloc[:,n_features+2:n_features+3]

#create Tensors to hold inputs and outputs, and wrap them in Variables,
#as Torch only trains nn on Variables (arrays)
X = Variable(torch.Tensor(train_input.values).float())
Y = Variable(torch.Tensor(train_target.values).long())

test_input = test_data.iloc[:,1:n_features+1]
test_target= test_data.iloc[:,n_features+2:n_features+3]

"""
Step 2: Define a neural network 

Here I build a neural network with one hidden layer.
    input layer: 16 neurons, representing the features of forests
    hidden layer: 13 neurons, using RELU as activation function
    output layer: 2 neurons, representing the type of glass

The network will be trained with Stochastic Gradient Descent (SGD) as an
optimiser, that will hold the current state and will update the parameters
based on the computed gradients.

Its performance will be evaluated using cross-entropy.
"""

# define the number of inputs, classes, training epochs, and learning rate
input_neurons = n_features
hidden_neurons = 13
output_neurons = 2
learning_rate = 0.01
num_epochs = 1000


# define a customised neural network structure
class TwoLayerNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(TwoLayerNet, self).__init__()
        # define linear hidden layer output
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        # define linear output layer output
        self.out = torch.nn.Linear(n_hidden, n_output)
        
        #self.coefficient = torch.nn.Parameter(torch.Tensor([1.3]))
    def forward(self, x):
        """
            In the forward function we define the process of performing
            forward pass, that is to accept a Variable of input
            data, x, and return a Variable of output data, y_pred.
        """
        # get hidden layer input
        h_input = self.hidden(x)
        # define activation function for hidden layer
        h_output = torch.relu(h_input)
        # get output layer output
        y_pred = self.out(h_output)

        return y_pred

# define a neural network using the customised structure
net = TwoLayerNet(input_neurons, hidden_neurons, output_neurons)
# define loss function
#loss_func = torch.nn.MSELoss()
loss_func = torch.nn.CrossEntropyLoss()

# define optimiser
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)


all_losses=[]

"""
Step 3: Train a neural network 

    To train the network, here setting errors once it is less than 70%
    change the weight matrix by multpling a small increment factor
    
"""

# train a neural network
for epoch in range(num_epochs):
    # Perform forward pass: compute predicted y by passing x to the model.
    # Here we pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    # In this case, Y_pred contains three columns, where the index of the
    # max column indicates the class of the instance
    Y_pred = net(X)
	
	# To suit the cross function 
    
    Y= Y.squeeze_()
    # Compute loss
    # Here we pass Tensors containing the predicted and true values of Y,
    # and the loss function returns a Tensor containing the loss.
    loss = loss_func(Y_pred, Y)
    all_losses.append(loss.item())

    # print progress
    if epoch % 50 == 0:
        # convert three-column predicted Y values to one column for comparison
        _, predicted = torch.max(F.softmax(Y_pred,1), 1)

        # calculate and print accuracy
        total = predicted.size(0)
        correct = predicted.data.numpy() == Y.data.numpy()
     
        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct)/total))
        if(100*sum(correct)/total<70):
            coefficient = torch.nn.Parameter(torch.Tensor([0.03]))
     
    # Clear the gradients before running the backward pass.
    net.zero_grad()

    # Perform backward pass: compute gradients of the loss with respect to
    # all the learnable parameters of the model.
    loss.backward()

    # Calling the step function on an Optimiser makes an update to its
    # parameters
    optimiser.step()

	#show the figure 
plt.figure()
plt.plot(all_losses)
plt.show()

# to check the weights matrix 
#for param in  net.parameters():
#    print(param.data)
"""
Step 4:Evaluating the Results by confusion matrix but not related to paper 

To see how well the network performs on different categories, we will
create a confusion matrix, indicating for every glass (rows)
which class the network guesses (columns).

"""

confusion = torch.zeros(output_neurons, output_neurons)

Y_pred = net(X)

_, predicted = torch.max(F.softmax(Y_pred,1),1)

for i in range(train_data.shape[0]):
    actual_class = Y.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for training:')
print(confusion)


"""
Step 5: Test the neural network

Pass testing data to the built neural network and get its performance
"""

# create Tensors to hold inputs and outputs
X_test = torch.tensor(test_input.values, dtype=torch.float)
Y_test = torch.tensor(test_target.values,dtype=torch.long)

# test the neural network using testing data
# It is actually performing a forward pass computation of predicted y
# by passing x to the model.
# Here, Y_pred_test contains three columns, where the index of the
# max column indicates the class of the instance
Y_pred_test = net(X_test)

# get prediction
# convert three-column predicted Y values to one column for comparison
_, predicted_test = torch.max(F.softmax(Y_pred_test, 1),1)

# calculate accuracy
total_test = predicted_test.size(0)
correct_test = np.sum(predicted_test.data.numpy()[1] == Y_test.data.numpy())
print('Testing Accuracy: %.2f %%' % (100*(correct_test / total_test)))



# In[ ]:




