# Kamangar, Farhad
# 1000_123_456
# 2020_10_30
# Assignment_03_01

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import math


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension=input_dimension
        self.weights=[]
        self.biases =[]
        self.activation_functions=[]
        self.loss=None
        self.multinn=[]


    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        self.weights.append(tf.zeros(shape=(self.input_dimension,num_nodes),dtype="float64"))
        self.input_dimension=num_nodes
        self.biases.append(tf.zeros(shape = ( 1, num_nodes), dtype = 'float64'))
        self.activation_functions.append(transfer_function)
        #self.weights=tf.Variable(tf.random.normal([self.input_dimension,num_nodes]))
        #self.biases=tf.Variable(tf.zeros([num_nodes]))
        self.multinn.append([tf.zeros(shape=(self.input_dimension,num_nodes),dtype="float64",name='w'),tf.zeros(shape = ( 1, num_nodes), dtype = 'float64',name='b'),transfer_function])

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return(self.weights[layer_number])
        #return self.multinn[layer_number][0]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return (self.biases[layer_number])

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number]=weights
        self.multinn[layer_number][0]=weights

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.biases[layer_number]=biases
        self.multinn[layer_number][1]=biases

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
    def Sigmoid(self, x):

        return tf.nn.sigmoid(x)

    def Linear(self, x):
        return x

    def Relu(self, x):
        out = tf.nn.relu(x)
        return out
    
    def one_hot_encodify(self, y):
        onehot_encoded_y_train = list()
        for value in y:
            out = [0 for _ in range(self.weights[-1].shape[1])]
            out[value] = 1
            onehot_encoded_y_train.append(out)
        onehot_encoded_y_train = np.array(onehot_encoded_y_train)
        return(onehot_encoded_y_train)
    
    def train_on_batch(self, x, y, learning_rate):
        with tf.GradientTape() as tape:
            predictions = self.predict(x)
            loss = self.loss(y, predictions)
            dloss_dw, dloss_db = tape.gradient(loss, [self.weights, self.biases])
        '''iterate over the layers'''
        for ((index, w), b) in zip(enumerate(self.weights), self.biases):
            w.assign_sub(learning_rate * dloss_dw[index])
            b.assign_sub(learning_rate * dloss_db[index])
        return None
    
    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        array_of_input=X
        f=0
        print(len(self.multinn))
        while(f<(len(self.multinn))):
            
            array_of_input=tf.matmul(array_of_input,self.multinn[f][0]) 
            array_of_input=tf.add(array_of_input,self.multinn[f][1])

            if self.multinn[f][2].lower()=='linear':
                array_of_input = array_of_input
                
            elif self.multinn[f][2].lower()=='sigmoid':
                array_of_input=tf.nn.sigmoid(array_of_input)

            elif self.multinn[f][2].lower()=='relu':
                array_of_input=tf.nn.relu(array_of_input)
            
            f=f+1
            
        return (array_of_input)

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        self.Y_train = tf.Variable(y_train)
        self.X_train = tf.Variable(X_train)

        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.batch(batch_size)

        for epoch in range(num_epochs):
            for step, (x, y) in enumerate(dataset):
                self.train_on_batch(x, y, alpha)



    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        self.X_train = tf.Variable(X)

        #one hot encoding
        self.Y_train = y

        #call predict
        self.Y_Predict = self.predict(X)

        #converting to lists
        Ytrain = tf.Variable(self.Y_train).numpy()
        Ytrain = Ytrain.tolist()
        YPredict = tf.Variable(self.Y_Predict).numpy()
        YPredict = YPredict.tolist()

        samples = 0
        mismatch_count = 0
        #iterating over the samples
        for (input_sample, T_Output, A_Output) in zip(self.X_train, Ytrain, YPredict):
            samples = samples + 1
            if np.argmax(T_Output) != np.argmax(A_Output):
                mismatch_count = mismatch_count + 1
        percent_error = (mismatch_count/samples)
        return(percent_error)

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        self.X_test=tf.Variable(X)
        self.Y_test=y
        self.Y_predict=self.predict(X).numpy()
        #self.Y_predict=self.Y_predict.flatten().astype(np.int32)
        print("testtt",self.Y_predict.shape)
        print("testtssss11",type(self.Y_test))
        
        #self.Y_test=self.Y_test.reshape(self.X_test.shape[0],self.Y_predict.shape[1])
        print("testtt1111",self.Y_test)
        #self.Y_test=np.argmax(self.Y_test)
        #self.Y_predict=np.argmax(self.Y_predict)
        rows,number_of_classes=np.shape(X)
        print(rows)
        
        confusion_matrix = np.zeros(shape = (self.weights[-1].shape[1], self.weights[-1].shape[1]))

        for row1, row2 in zip(self.Y_test, self.Y_predict):
            i = row1
            j = np.argmax(row2)
            confusion_matrix[i][j] += 1
        print(confusion_matrix)
        return(confusion_matrix)
