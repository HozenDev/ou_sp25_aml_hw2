from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1, l2

# Define an improved deep_network_basic function
def deep_network_basic(n_inputs:int,
                       n_hidden:list,
                       n_output:int,
                       activation:str='elu',
                       activation_out:str='linear',
                       dropout:float=0.0,
                       dropout_input:float=0.0,
                       kernel_regularizer:float=0.0,
                       kernel_regularizer_L1:float=0.0,
                       metrics=['mse'],
                       lrate:float=0.001) -> Sequential:
    """
    Constructs a sequential neural network model.

    :param n_inputs: Number of input features.
    :param n_hidden: List containing the number of neurons for each hidden layer.
    :param n_output: Number of output neurons.
    :param activation: Activation function for the hidden layers.
    :param activation_out: Activation function for the output layer.
    :param dropout: Dropout rate for hidden layers.
    :param dropout_input: Dropout rate for input layer.
    :param kernel_regularizer: L2 regularization parameter.
    :param kernel_regularizer_L1: L1 regularization parameter.
    :param metrics: List of metrics to track during training.
    :param lrate: Learning rate for the optimizer.
    :return: Compiled Keras sequential model.
    """
    model = Sequential()
    model.add(InputLayer(shape=(n_inputs,)))

    # Dropout on INPUT layer
    if dropout_input is not None and dropout_input > 0.0:
        model.add(Dropout(dropout_input))

    # Hidden layer management
    for i, n in enumerate(n_hidden):
        model.add(Dense(n, activation=activation, name=f'hidden{i}'))
        if dropout is not None and dropout > 0.0:
            model.add(Dropout(dropout))

    # Output layer management
    if kernel_regularizer is not None and kernel_regularizer > 0.0:
        model.add(Dense(n_output, activation=activation_out, 
                        name='output', kernel_regularizer=l2(kernel_regularizer)))
    elif kernel_regularizer_L1 is not None and kernel_regularizer_L1 > 0.0:
        model.add(Dense(n_output, activation=activation_out, 
                        name='output', kernel_regularizer=l1(kernel_regularizer_L1)))
    else:
        model.add(Dense(n_output, activation=activation_out, name='output'))

    # Compile the model
    opt = Adam(learning_rate=lrate)
    model.compile(loss='mse', optimizer=opt, metrics=metrics)
    
    return model
