import numpy as np
import tensorflow as tf
import tensorflow.keras.activations as af

# implement mish activation function
def f_mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    '''
    return input * af.tanh(af.softplus(input))

# implement class wrapper for mish activation function
class mish(tf.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return f_mish(input)




# implement TAct activation function

def f_tact(input, alpha, beta, inplace = False):
    '''
    Applies the tact function element-wise:
    tact(x) = ----
    '''
    A = 0.5*alpha
    B = 0.5 - A
    #B=(1-alpha)/2
    C = 0.5*(1+beta)

    return (A*input + B)*(af.tanh(C*input)+1)

# implement class wrapper for TAct activation function
class TACT(tf.Module):
    '''
    Applies the TACT function element-wise:
    tact(x) = ----

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> t = tact()
        >>> input = torch.randn(2)
        >>> output = t(input)

    '''
    def __init__(self, alpha = np.random.uniform(0,0.5), beta = np.random.uniform(0,0.5),  inplace = False):
        """
        An implementation of our M Tanh Activation Function,
        mTACT.
        :param alpha a tuneable parameter
        :param beta a tuneable parameter
        """
        super().__init__()
        self.inplace = inplace

        self.alpha = alpha
        self.alpha = tf.Variable(self.alpha)

        self.beta = beta
        self.beta = tf.Variable(self.beta)

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return f_tact(input, alpha = self.alpha, beta = self.beta, inplace = self.inplace)



# implement MTAct activation function

def f_mtact(input, alpha, beta, inplace = False):
    '''
    Applies the mtact function element-wise:
    mtact(x) = ----
    '''
    A = 0.5*(alpha**2)
    B = 0.5 - A
    #B=(1-alpha**2)/2
    #C = (1+beta**2)/2
    C = 0.5*(1+beta**2)

    return (A*input + B)*(af.tanh(C*input)+1)

# implement class wrapper for MTAct activation function
class mTACT(tf.Module):
    '''
    Applies the mTACT function element-wise:
    mtact(x) = ----

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = mtact()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self, alpha = np.random.uniform(0,0.5), beta = np.random.uniform(0,0.5),  inplace = False):
        """
        An implementation of our M Tanh Activation Function,
        mTACT.
        :param alpha a tuneable parameter
        :param beta a tuneable parameter
        """
        super().__init__()
        self.inplace = inplace

        self.alpha = alpha
        self.alpha = tf.Variable(self.alpha)

        self.beta = beta
        self.beta = tf.Variable(self.beta)

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return f_mtact(input, alpha = self.alpha, beta = self.beta, inplace = self.inplace)
