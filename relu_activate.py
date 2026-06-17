import numpy as np
class ReluActivation:   
 
    def relu(self,previous_layer_output):
        inputLayerDerivative = (previous_layer_output > 0).astype(float)
        inputLayerOutput = np.maximum(0, previous_layer_output)
        return inputLayerOutput,inputLayerDerivative
