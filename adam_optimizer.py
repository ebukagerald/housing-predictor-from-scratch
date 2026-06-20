import numpy as np
# _________________ADAM OPTIMIZER___________________1,134,34,56,90,1,9,8
              
class AdamOptimizer:
    def __init__(self):
        
        self.hyperParameter1 = 0.9000
        self.hyperParameter2 = 0.9990
        self.smoothening = 1e-8
        self.learningRate =0.01

    
    def adam_optimizer(self,adam_opt_dict):
    
        grad = adam_opt_dict['grad']
        oldWeightMatrix = adam_opt_dict['oldWeightMatrix']
        ith = adam_opt_dict['ith']
        lastLayer_firstMoment = adam_opt_dict['lastLayer_firstMoment']
        lastLayer_secondMoment = adam_opt_dict['lastLayer_secondMoment']

        m1 = self.hyperParameter1 * lastLayer_firstMoment + (1 - self.hyperParameter1)*grad
        v1 = self.hyperParameter2 * lastLayer_secondMoment+ (1 - self.hyperParameter2 )*grad**2
    
        firstMomentBias = m1/(1-self.hyperParameter1**ith)
        secondMomentBias = v1/(1-self.hyperParameter2**ith)
        
        newWeight = oldWeightMatrix - self.learningRate * firstMomentBias / (np.sqrt(secondMomentBias) + self.smoothening) 
        return newWeight , m1, v1

