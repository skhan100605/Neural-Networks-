# Khan, Sahiba
# 1002_083_293
# 2022_09_25
# Assignment_01_01

import numpy as np

class SingleLayerNN(object):
    def __init__(self, input_dimensions=2,number_of_nodes=4):
        """
        Initialize SingleLayerNN model and set all the weights and biases to random numbers.
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: Note that number of neurons in the model is equal to the number of classes.
        """
        self.input_dimensions=input_dimensions
        self.number_of_nodes=number_of_nodes
        self.initialize_weights()
        self.output=[]
    def initialize_weights(self,seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
            self.weights=np.random.randn(self.number_of_nodes,self.input_dimensions+1)
        else:
            self.weights=np.ones((self.number_of_nodes,self.input_dimensions+1))

    def set_weights(self, W):
        """
        This function sets the weight matrix (Bias is included in the weight matrix).
        :param W: weight matrix
        :return: None if the input matrix, w, has the correct shape.
        If the weight matrix does not have the correct shape, this function
        should not change the weight matrix and it should return -1.
        """
        shape_of_weights=self.weights.shape
        print(W)
        if(shape_of_weights[0] != self.input_dimensions and shape_of_weights[1] != self.number_of_nodes):
            return -1
        else:
            self.weights=W

    def get_weights(self):
        """
        This function should return the weight matrix(Bias is included in the weight matrix).
        :return: Weight matrix
        """
        return self.weights
    def hardlimit(self,net):
        if(net <0):
           return 0
        else:
            return 1

    def predict(self, X):
        """
        Make a prediction on a batach of inputs.
        :param X: Array of input [input_dimensions,n_samples]
        :return: Array of model [number_of_nodes ,n_samples]
        Note that the activation function of all the nodes is hard limit.
        """
        
        self.out_arr = np.zeros([self.number_of_nodes, X[0].size], int)
        #transpose
        self.out_arr = self.out_arr.T
        self.X_training = np.concatenate((np.ones([1,X[0].size]),X))

        self.X_training = self.X_training.T
        #print("train",self.X_training)


        for (input_sample, output) in zip(self.X_training, self.out_arr):
            #iterating over each sample
            for i in range(self.number_of_nodes):
                prediction = None
                net = 0
                for (xi, wi) in zip(input_sample, self.weights[i]):
                    #finding net value
                    net = net + (xi * wi)
                    #print("net value here: ",net)
                
                prediction=self.hardlimit(net)
                output[i] = prediction

        return(self.out_arr.T)
       


    def train(self, X, Y, num_epochs=10,  alpha=0.1):
        """
        Given a batch of input and desired outputs, and the necessary hyperparameters (num_epochs and alpha),
        this function adjusts the weights using Perceptron learning rule.
        Training should be repeated num_epochs times.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        #training the samples
        for epoch in range(num_epochs):
            input = X
            self.out_arr = self.predict(input).T

            self.given_out = Y.T

            self.X_training = np.concatenate((np.ones([1,X[0].size]),X))
            self.X_training = self.X_training.T

            for (input_sample, target_out, target_out_row) in zip(self.X_training, self.given_out, range(len(self.given_out))):
                actual_out = self.out_arr[target_out_row]
                for i in range(self.number_of_nodes):
                    #finding error 
                    error = target_out[i] - actual_out[i]
                    #print("err",error)
                    for (input, weight, weight_index) in zip(input_sample, self.weights[i], range(len(self.weights[i]))):
                        #updating the weights
                        self.weights[i][weight_index] = weight + (alpha * error * input)
                input = X
                self.out_arr = self.predict(input).T


    def calculate_percent_error(self,X, Y):
        """
        Given a batch of input and desired outputs, this function calculates percent error.
        For each input sample, if the output is not the same as the desired output, Y,
        then it is considered one error. Percent error is 100*(number_of_errors/ number_of_samples).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :return percent_error
        """
        self.X_training = np.concatenate((np.ones([1,X[0].size]),X))
        self.X_training = self.X_training.T
        self.given_out = Y.T
        #call predict
        input = X
        self.out_arr = self.predict(input).T
        samples = 0
        count = 0
        for (input_sample, target_out, actual_out) in zip(self.X_training, self.given_out, self.out_arr):
            samples = samples + 1
            #iterate over the predictions made by each i
            for i in range(self.number_of_nodes):
                #print("ti", target_out[i])
                #print("ai", actual_out[i])
                if target_out[i] != actual_out[i]:
                    count = count + 1
                    break
        percent_error = (count/samples)*100
        print(percent_error)
        return(percent_error)


if __name__ == "__main__":
    input_dimensions = 2
    number_of_nodes = 2

    model = SingleLayerNN(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=2)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    print("****** Model weights ******\n",model.get_weights())
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.1)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.get_weights())