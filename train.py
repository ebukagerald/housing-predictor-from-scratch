import numpy as np
import pandas as pd
import math
import logging
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from backward_propagation import BackwardPropagation
logging.basicConfig(level=logging.ERROR)
from dataset_processing import DatasetProcess, DatasetError

ith = 1
batch_counter=0
totalLoss = 0
batch_size = 5
input_sample = 14
no_of_first_hidden_neuron = 8
no_of_output_neuron = 4

backwardprop = BackwardPropagation(ith, batch_counter)
lastLayerDecent = backwardprop.lastLayerDecent

forward_input_class = backwardprop.forward_input_class
forward_hidden_class = backwardprop.forward_hidden_class
forward_output_class = backwardprop.forward_output_class


forward_input_class.neuron_weights = np.random.randn(no_of_first_hidden_neuron ,input_sample) * np.sqrt(2/input_sample)
forward_input_class.neuron_bias = np.zeros(no_of_first_hidden_neuron ) 


forward_hidden_class.neuron_weights = np.random.randn(no_of_output_neuron,no_of_first_hidden_neuron ) * np.sqrt(2/no_of_first_hidden_neuron )
forward_hidden_class.neuron_bias = np.zeros(no_of_output_neuron) 

forward_output_class.neuron_weights_output = np.random.randn(1,no_of_output_neuron) * np.sqrt(2/no_of_output_neuron)
forward_output_class.neuron_bias_output  = np.zeros(1)
        

filename = "/home/ebuka/deeplearning/Neural-Network-From-Scratch/dataset/Housing.csv"

try:
    data_class = DatasetProcess(filename,"price" ,0.8) 
    X_train, y_train , X_test, y_test,column_names = data_class.transform_columns()
except DatasetError as e:
    print({e})
    
def sumTarget(y_target, y_pred):
    return np.mean((y_target - y_pred) ** 2)


forward_input_inst  = forward_input_class.input_layer_one
forward_hidden_inst = forward_hidden_class.input_layer_one
forward_output_inst = forward_output_class.output_layer

total_loss = []
for epoch in range(1000):
    totalLoss = 0
    batch_counter=0
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    for i in range(len(y_train)):
        batch_counter +=1
        backwardprop.batch_counter = batch_counter
        
        y_pred = forward_output_inst(forward_hidden_inst(forward_input_inst(X_train[i])))
            
        totalLoss += sumTarget(y_train[i], y_pred)
        lastLayerDecent( y_pred,y_train[i],X_train[i],len(y_train),batch_size)

    if epoch % 100 == 0:
        print("Epoch:", epoch, "Loss:", totalLoss / len(y_train))
        total_loss.append(totalLoss / len(y_train))
        # print(len(y_train))

print("=" * 50)
testLoss = 0
ss_res =0
ss_tot = 0
y_test_mean = np.mean(y_test)
predictions = []

for i in range(len(y_test)):
    y_pred = forward_output_inst(forward_hidden_inst(forward_input_inst(X_test[i])))
    predictions.append(y_pred)
    testLoss += sumTarget(y_test[i], y_pred)
    ss_res += np.sum((y_test[i] - y_pred)**2)
    ss_tot += np.sum((y_test[i] - y_test_mean )**2)

predictions = np.array(predictions)
rmse = np.sqrt(np.mean((y_test - predictions)**2))
mae = np.mean(np.abs(y_test - predictions))
r2 = 1 - (ss_res / ss_tot)

print(f"Test Loss: {testLoss / len(y_test)}")
print("=" * 50)
print(f"R-Square:{r2}")
print("=" * 50)
print(f"rmse :{rmse }")
print("=" * 50)
print(f"mae :{mae}")

plt.plot(total_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig('my_plot.png') 


