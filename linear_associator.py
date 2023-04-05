# Khan, Sahiba
# 1002_083_293
# 2022_10_09
# Assignment_02_01

import numpy as np
import math

class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_dimensions=input_dimensions
        self.number_of_nodes=number_of_nodes
        self.transfer_function=transfer_function

        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """

        if(seed != None):
            np.random.seed(seed)
            self.weights=np.random.randn(self.number_of_nodes,self.input_dimensions)
        else:
            self.weights=np.ones((self.number_of_nodes,self.input_dimensions),dtype=float, order='C')

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        if(self.weights.shape[0] != self.input_dimensions and self.weights.shape[1] != self.number_of_nodes):
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
    def linear(self,net):
        return net

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        #creating o/p array
        self.Y_out_arr=np.zeros([self.number_of_nodes, X[0].size]).T
        self.X_for_training=X
        self.X_for_training=self.X_for_training.T
        for(sample,output) in zip(self.X_for_training,self.Y_out_arr):
            for i in range(self.number_of_nodes):
                prediction=None
                net=0
                for(input,weight) in zip(sample,self.weights[i]):
                    
                    net = net + (input*weight)
                    if(self.transfer_function=='Hard_limit'):
                        prediction=self.hardlimit(net)
                    else:
                        prediction=self.linear(net)
                output[i]=prediction
        return self.Y_out_arr.T

    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        #W=TP+ T=y P=X
        #calculating p+
        self.P_T=X.T
        self.P=X
        self.P_temp=np.dot(self.P_T,self.P)
        self.P_temp_2=np.linalg.pinv(self.P_temp)
        self.P_temp3=np.dot(self.P_temp_2,self.P.T)
        #self.P_plus_value=np.dot(np.linalg.inv(np.dot(self.P_T,self.P)),self.P_T)
        self.P_plus_value=self.P_temp3
        self.weights=np.dot(y,self.P_plus_value)

    def delta(self,W,alpha,error,sample):
        #print("del err",error.shape)
        #print("W",W.shape)
        return np.add(W, (np.dot(error.T,sample)*alpha))
        #return W
    
    def filtered(self,W,alpha,gamma,target_out,sample):
        return np.add(((1-gamma)*W), (np.dot(target_out.T,sample)*alpha))

    def unsupervised_hebb(self,W,alpha,actual_out,sample):
        #print("sam",sample.shape)
        return np.add(W,(np.dot(actual_out.T,sample)*alpha))


    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        for epoch in range(num_epochs):
            input_for_predict=X
            self.Y_predict_out=self.predict(input_for_predict).T
            #print("predict",self.Y_predict_out.shape)
            self.Y_train=y.T
            self.X_train=X.T
            #print("X train",self.X_train.shape)

            number_of_batches=math.ceil(X.shape[1]/batch_size)
            lower_index=0
            upper_index=batch_size
            for i in range(number_of_batches):
                #print("low here",lower_index)
                #print("upper here",upper_index)
                sample_train_input=self.X_train[lower_index:upper_index]
                #print("sample",sample_train_input.shape)
                target_out=self.Y_train[lower_index:upper_index]
                #print("target",target_out.shape)
                #print("pre",self.Y_predict_out.shape)
                actual_out=self.Y_predict_out[lower_index:upper_index]
                #print("actual",actual_out.shape)
                error=np.subtract(target_out,actual_out)
                #print("error",error.shape)
                if(learning=="delta" or learning=="Delta"):
                    self.weights=self.delta(self.weights,alpha,error,sample_train_input)
                elif(learning == "Filtered"):
                    self.weights=self.filtered(self.weights,gamma,alpha,target_out,sample_train_input)
                else:
                    self.weights=self.unsupervised_hebb(self.weights,alpha,actual_out,sample_train_input)
                input=X
                self.Y_predict_out=self.predict(input).T
                lower_index=lower_index+batch_size
                #print("low",lower_index)
                upper_index=upper_index+batch_size
                #print("upper",upper_index)
        #print("y",y.shape)
        # for epoch in range(num_epochs):
        #     input_req=X
        #     #print("inp",input_req.shape)
        #     self.Y_predict_out_train=self.predict(input_req).T
        #     #print("predict",self.Y_predict_out_train.shape)
        #     self.Y_train=y.T
        #     #print("y",self.Y_train.shape)
        #     self.X_train=X.T
        #     number_of_batches=math.ceil(self.X_train.shape[1]/batch_size)
        #     #print("batch",batches_req)
        #     l=0
        #     u=batch_size
        #     for batch in range(number_of_batches):
        #         #print("l",l)
        #         #print("u",u)
        #         input_req_batch=self.X_train[l:u]
        #         target_out=self.Y_train[l:u]
        #         #print("target_out",target_out.shape)
        #         actual_out=self.Y_predict_out_train[l:u]
        #         #print("actual_out",actual_out.shape)
        #         error=np.subtract(target_out,actual_out)
        #         if(learning=="Delta"):
        #             self.weights=self.delta(self.weights,alpha,error,input_req_batch)
        #         elif(learning == 'Filtered'):
        #             self.weights=self.filtered(self.weights,alpha,gamma,target_out,input_req_batch)
        #         elif(learning == "Unsupervised_hebb"):
        #             self.weights=self.unsupervised_hebb(self.weights,alpha,actual_out,input_req_batch)
        #         input_new=X
        #         self.Y_predict_out_train=self.predict(input_new).T
        #         l=l+batch_size
        #         u=u+batch_size


    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        self.X_train=X.T
        self.Y_train=y.T
        input=X
        self.Y_predict_out=self.predict(input).T

        mse = np.mean((self.Y_train - self.Y_predict_out)**2)
        return mse

                
        
