'''
This is one of the modules you'll submit to the autograder. The positional encoding is implemented in this file.
'''

'''
Note:
Please do not modify any variable name given to you for code completion, especially those that have trainable parameters in torch
'''

import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    '''
    We implement the positional encoding as alternating sin and cosine functions.
    In the __init__ function, you simply set the variable pe according the original paper.
    In the forward function, you add the pe to the input
    '''
    def __init__(self, d_model, max_seq_length):
        '''
        Initialize the positional encoding as alternating sin and cosine functions of the dimension position and the model size d_model
        Input:
            d_model (int) - dimension of the Transformer
            
            max_seq_length (int) - maximum sequence length of the Transformer
        '''
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)

        ## fill in the following code for positional encodings
        ## note that for better numerical accuracy, (1/10000) ^ (2i / d_model) = exp(2i * (-log(10000) /d_model))

        ##### YOUR CODE STARTS HERE #####

        # find the position and denominator tensors and match them for multiplication
        pos = torch.arange(0, max_seq_length, 1).unsqueeze(1)
        den = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000))/d_model).unsqueeze(0)
        # set pe[pos, 2i]
        pe[:, 0::2] = torch.sin(pos * den)
        # set pe[pos, 2i + 1]
        pe[:, 1::2] = torch.cos(pos * den)
 
        ##### YOUR CODE ENDS HERE #####

        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """Add positional encoding (after appropriate slicing) to the input x

        Input:
            x (torch.Tensor) - Tensor of size B x T x d_model.

        Output:
            torch.Tensor - input with positional encoding added

        """
        ##### YOUR CODE STARTS HERE #####
        
        # slice pe so it becomes (1, T, d_model)
        return x + self.pe[:, :x.size()[1], :]
    
        ##### YOUR CODE ENDS HERE #####