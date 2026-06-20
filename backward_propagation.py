import numpy as np
from layers import ForwardPropagation
from adam_optimizer import AdamOptimizer


class BackwardPropagation:
    def __init__(self, ith, batch_counter):
        self.forward_input_class = ForwardPropagation()
        self.forward_hidden_class = ForwardPropagation()
        self.forward_output_class = ForwardPropagation()
        
        self.adam_intc = AdamOptimizer()
        self.adam_optimizer = self.adam_intc.adam_optimizer
        self.ith = ith
        self.batch_counter = batch_counter
        self.intialize_parameter = 0
        self.just_testing =None
        self.update = False
        self.samples_remaining = None
        self.batch_size = None
        self.train_len = None

        
    def initialize_parameters(self):         
        
        self.batch_gradient_last_layer = np.zeros_like(self.forward_output_class.neuron_weights_output)
        self.batch_gradient_middle_layer = np.zeros_like(self.forward_hidden_class.neuron_weights)
        self.batch_gradient_input_layer = np.zeros_like(self.forward_input_class.neuron_weights)
        self.batch_gradient_bias_middle_layer=np.zeros_like(self.forward_hidden_class.neuron_bias)
        self.batch_gradient_bias_input_layer = np.zeros_like(self.forward_input_class.neuron_bias )
        self.gradient_last_layer_l_y = np.zeros_like(self.forward_output_class.neuron_bias_output)
        
        self.lastLayer_firstMomentVector = np.zeros_like(self.forward_output_class.neuron_weights_output)
        self.lastLayer_secondMomentVector = np.zeros_like(self.forward_output_class.neuron_weights_output)
        self.lastLayerBias_firstMomentVector = np.zeros_like(self.forward_output_class.neuron_bias_output)
        self.lastLayerBias_secondMomentVector = np.zeros_like(self.forward_output_class.neuron_bias_output)
        
        self.middleLayer_firstMomentVector = np.zeros_like(self.forward_hidden_class.neuron_weights)
        self.middleLayer_secondMomentVector = np.zeros_like(self.forward_hidden_class.neuron_weights)
        self.middleLayerBias_firstMomentVector = np.zeros_like(self.forward_hidden_class.neuron_bias)
        self.middleLayerBias_secondMomentVector = np.zeros_like(self.forward_hidden_class.neuron_bias)
        
        self.inputLayer_firstMomentVector = np.zeros_like(self.forward_input_class.neuron_weights)
        self.inputLayer_secondMomentVector = np.zeros_like(self.forward_input_class.neuron_weights)
        self.inputLayerBias_firstMomentVector = np.zeros_like(self.forward_input_class.neuron_bias)
        self.inputLayerBias_secondMomentVector = np.zeros_like(self.forward_input_class.neuron_bias)
            
                
    def lastLayerDecent(self,y_pred, y_train,X,train_len,batch_size ):
        if self.intialize_parameter == 0:
            self.initialize_parameters()
            self.intialize_parameter +=1
            self.batch_size=batch_size
            self.train_len =  train_len
            self.samples_remaining =  train_len % batch_size
            
        ith = self.ith
        batch_counter = self.batch_counter   
        forward_output_class = self.forward_output_class
        forward_input_class = self.forward_input_class
        forward_hidden_class = self.forward_hidden_class
        adam_optimizer = self.adam_optimizer
        
        batch_gradient_last_layer = self.batch_gradient_last_layer
        gradient_last_layer_l_y = self.gradient_last_layer_l_y
        lastLayer_firstMomentVector = self.lastLayer_firstMomentVector
        lastLayer_secondMomentVector  = self.lastLayer_secondMomentVector
        lastLayerBias_firstMomentVector = self.lastLayerBias_firstMomentVector
        lastLayerBias_secondMomentVector = self.lastLayerBias_secondMomentVector
        
    
        neuron_weights_final_output = forward_output_class.neuron_weights_output
        neuron_bias_final_output = forward_output_class.neuron_bias_output 
        hidden_layer_one_output = forward_hidden_class.input_layer_output
        
        oldWeight = neuron_weights_final_output.copy()
        change_L_Y = np.array([2 * (y_pred - y_train)])
        change_L_W =  np.outer(change_L_Y,hidden_layer_one_output )    # shape (8,)
    
    # summing gradients for mini-batch
        gradient_last_layer_l_y +=change_L_Y 
        batch_gradient_last_layer += change_L_W
        
        if batch_counter > 0 and batch_counter % self.batch_size == 0:
            change_L_W = batch_gradient_last_layer /self.batch_size
            change_L_Y = gradient_last_layer_l_y /self.batch_size
            self.update = True

        elif batch_counter ==  self.train_len :  
            change_L_W = batch_gradient_last_layer / self.samples_remaining 
            change_L_Y = gradient_last_layer_l_y / self.samples_remaining 
            self.update = True

        if self.update:                                                               
            self.batch_gradient_last_layer = np.zeros_like(neuron_weights_final_output)
            self.gradient_last_layer_l_y = np.zeros_like(neuron_bias_final_output)
    
            
            adam_weight_dict = {'grad': change_L_W,'oldWeightMatrix':neuron_weights_final_output,'ith':ith,
                               'lastLayer_firstMoment':lastLayer_firstMomentVector,
                               'lastLayer_secondMoment':lastLayer_secondMomentVector}
           
            adam_bias_dict = {'grad': change_L_Y,'oldWeightMatrix':neuron_bias_final_output,'ith':ith,
                               'lastLayer_firstMoment':lastLayerBias_firstMomentVector,
                               'lastLayer_secondMoment':lastLayerBias_secondMomentVector}
            
            self.forward_output_class.neuron_weights_output, self.lastLayer_firstMomentVector, self.lastLayer_secondMomentVector  = adam_optimizer(adam_weight_dict)
            self.forward_output_class.neuron_bias_output  , self.lastLayerBias_firstMomentVector,  self.lastLayerBias_secondMomentVector = adam_optimizer(adam_bias_dict  )
            
            
            # neuronWeights_Final_Output -= 0.001 * change_L_W 
            # neuronBias_Final_Output -= 0.001 * change_L_Y
            
            
    # Called the middleLayerDescent function
        # print(self.ith)
        self.middleLayerDescent(change_L_Y,X,oldWeight)
        
        
          
    
    def middleLayerDescent(self,change_L_Y,X,oldWeight):
        forward_output_class =  self.forward_output_class
        forward_input_class =self.forward_input_class
        forward_hidden_class = self.forward_hidden_class

        adam_optimizer = self.adam_optimizer
        
        batch_gradient_middle_layer = self.batch_gradient_middle_layer
        batch_gradient_bias_middle_layer = self.batch_gradient_bias_middle_layer
        batch_counter = self.batch_counter
        middleLayer_firstMomentVector = self.middleLayer_firstMomentVector
        middleLayer_secondMomentVector = self.middleLayer_secondMomentVector
        middleLayerBias_firstMomentVector  = self.middleLayerBias_firstMomentVector 
        middleLayerBias_secondMomentVector = self.middleLayerBias_secondMomentVector
        ith = self.ith
    
        neuron_weights_final_output = forward_output_class.neuron_weights_output
        neuron_weights_hidden_layer_1 = forward_hidden_class.neuron_weights 
        input_layer_output = forward_input_class.input_layer_output
        neuron_bias_hidden_layer_1 = forward_hidden_class.neuron_bias 
        hidden_layer_one_derivative = forward_hidden_class.input_layer_derivative  
        
        oldHiddenWeights = neuron_weights_hidden_layer_1.copy()
        delta2 =    oldWeight.T @ change_L_Y 
        delta2 = delta2 * hidden_layer_one_derivative
        grad = np.outer(delta2 , input_layer_output)
    
    # summing gradients for mini-batch
        batch_gradient_middle_layer += grad
        batch_gradient_bias_middle_layer +=delta2
        
        if batch_counter > 0 and batch_counter % self.batch_size == 0:
            grad = batch_gradient_middle_layer / self.batch_size
            delta2 = batch_gradient_bias_middle_layer / self.batch_size
            self.update = True
            

        elif batch_counter ==  self.train_len :  
            grad = batch_gradient_middle_layer / self.samples_remaining 
            delta2 = batch_gradient_bias_middle_layer / self.samples_remaining 
            self.update = True
                                                                              
        if self.update:
            self.batch_gradient_middle_layer = np.zeros_like(neuron_weights_hidden_layer_1)
            self.batch_gradient_bias_middle_layer=np.zeros_like(neuron_bias_hidden_layer_1)                                                                
        
            # neuronWeights_hiddenLayer_1  -= 0.001 * grad
            # neuronBias_hiddenLayer_1 -= 0.001 * delta2
    
            adam_weight_dict = {'grad': grad,'oldWeightMatrix':neuron_weights_hidden_layer_1,'ith':ith,
                               'lastLayer_firstMoment':middleLayer_firstMomentVector,
                               'lastLayer_secondMoment':middleLayer_secondMomentVector}
            adam_bias_dict = {'grad': delta2,'oldWeightMatrix':neuron_bias_hidden_layer_1,'ith':ith,
                               'lastLayer_firstMoment':middleLayerBias_firstMomentVector,
                               'lastLayer_secondMoment':middleLayerBias_secondMomentVector}
    
            self.forward_hidden_class.neuron_weights, self.middleLayer_firstMomentVector, self.middleLayer_secondMomentVector  = adam_optimizer(adam_weight_dict  )
            self.forward_hidden_class.neuron_bias , self.middleLayerBias_firstMomentVector,  self.middleLayerBias_secondMomentVector = adam_optimizer( adam_bias_dict )
            
    
        self.firstLayerDescent(delta2,X,oldHiddenWeights)
       
        # return neuronWeights_hiddenLayer_1 
    
    def firstLayerDescent(self,delta2, inputX_j,oldWeight):
        forward_input_class = self.forward_input_class
        forward_hidden_class = self.forward_hidden_class
        adam_optimizer = self.adam_optimizer
        batch_gradient_input_layer = self.batch_gradient_input_layer 
        batch_gradient_bias_input_layer = self.batch_gradient_bias_input_layer
        batch_counter = self.batch_counter
        inputLayer_firstMomentVector = self.inputLayer_firstMomentVector
        inputLayer_secondMomentVector = self.inputLayer_secondMomentVector
        inputLayerBias_firstMomentVector  =  self.inputLayerBias_firstMomentVector 
        inputLayerBias_secondMomentVector =  self.inputLayerBias_secondMomentVector
        ith= self.ith
    
        neuron_weights = forward_input_class.neuron_weights
        neuron_bias = forward_input_class.neuron_bias  
        input_layer_derivative = forward_input_class.input_layer_derivative  
    
        inputX_j = np.array(inputX_j)
        delta1 = oldWeight.T @ delta2
        delta1 = delta1 * input_layer_derivative
        grad1 = np.outer(delta1, inputX_j)
        
    # summing gradients for mini-batch
        batch_gradient_input_layer += grad1 
        batch_gradient_bias_input_layer += delta1
        
        if batch_counter > 0 and batch_counter % self.batch_size == 0:
    
            grad1 = batch_gradient_input_layer / self.batch_size
            delta1 = batch_gradient_bias_input_layer / self.batch_size
            self.update = True
            

        elif batch_counter ==  self.train_len :  
            grad1 = batch_gradient_input_layer  / self.samples_remaining 
            delta1 = batch_gradient_bias_input_layer / self.samples_remaining 
            self.update = True
                                                                              
        if self.update:
            self.batch_gradient_input_layer = np.zeros_like(neuron_weights)
            self.batch_gradient_bias_input_layer = np.zeros_like(neuron_bias ) 
            self.update = False
            # neuronWeights -= 0.000011 * grad1
            # neuronBias -= 0.001 * delta1
    
            adam_weight_dict = {'grad': grad1,'oldWeightMatrix':neuron_weights,'ith':ith,
                               'lastLayer_firstMoment':inputLayer_firstMomentVector,
                               'lastLayer_secondMoment':inputLayer_secondMomentVector}
            adam_bias_dict = {'grad': delta1,'oldWeightMatrix':neuron_bias,'ith':ith,
                               'lastLayer_firstMoment':inputLayerBias_firstMomentVector,
                               'lastLayer_secondMoment':inputLayerBias_secondMomentVector}
      
            self.forward_input_class.neuron_weights, self.inputLayer_firstMomentVector, self.inputLayer_secondMomentVector  = adam_optimizer(adam_weight_dict )
            self.forward_input_class.neuron_bias , self.inputLayerBias_firstMomentVector,  self.inputLayerBias_secondMomentVector = adam_optimizer(adam_bias_dict )
            self.ith +=1 
            
