import numpy as np
from relu_activate import ReluActivation

class ForwardPropagation:
    def __init__(self):
        self.neuron_weights = None
        self.neuron_bias =None
        self.neuron_weights_output = None
        self.neuron_bias_output  =None
        self.input_layer_output = None
        self.input_layer_derivative = None
       
    def input_layer_one(self,sample_input):   
        
        x = np.array(sample_input)             # shape: (14,)
        W = np.array(self.neuron_weights)       # shape: (16, 14)
        b = np.array(self.neuron_bias)          # shape: (16,)
        z = W @ x + b                     # matrix multiplication + bias shape: (16,)
        
        forward = ReluActivation()
        self.input_layer_output,self.input_layer_derivative = forward.relu(z)

        return  self.input_layer_output
       
        
    def output_layer(self,input_value):
        
        x = np.array(input_value)            # shape: (8,)
        W = np.array(self.neuron_weights_output  )
        b = np.array(self.neuron_bias_output )         # shape: (1,8)

        # print(x.size,W.size,b.size)
        z = W @ x + b
        
        return z.item()

