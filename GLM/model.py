import tensorflow as tf
import numpy as np

class GLMBase(object):
    
    def __init__(self, x_train, y_train,
                 activation=None,
                 loss=tf.losses.mean_squared_error,
                 optim=tf.train.AdamOptimizer(.1), 
                 dropout= None,
                 weight_regularization = None):
            
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("training and target data sample numbers must match")
            
        self._n_samples = x_train.shape[0]
        self._fit = False
        self._session = None
        self._x_train = x_train
        self._y_train = y_train
        self.dropout = dropout     
        self._activation = activation
        
        self._loss = loss
        self._optim = optim
              
        if self._x_train.ndim == 1:
            x_dim = 1
        else:
            x_dim = self._x_train.shape[1]
        if self._y_train.ndim == 1:
            y_dim = 1
        else:
            y_dim = self._y_train.shape[1]
            
        self._W_init = tf.truncated_normal(shape=(x_dim,y_dim))
        self._b_init = tf.truncated_normal(shape=(y_dim,))
        
        self._W = tf.Variable(self._W_init)
        self._b = tf.Variable(self._b_init)
        
        self._x = tf.placeholder(tf.float32)
        self._y = tf.placeholder(tf.float32)

        self._y_lin = (tf.matmul(self._x, self._W)) + self._b
        
        if self._activation is not None:
            self._y_pred = self._activation(self._y_lin) 
        else:
            self._y_pred = self._y_lin
            
        self._l = self._loss(self._y, self._y_pred)
        self._train = self._optim.minimize(self._l)
    
    def fit(self, iteration="epoch",
            n_steps=100, 
            minibatches=False, 
            batch_size=.1):
        
        #create session
        if not self._fit:
            self._session = tf.Session()
            self._session.run(tf.global_variables_initializer())
        
       
        #TODO batch training
        #batch training goes here
        
        #gradient descent
        #TODO early stopping
        feed = {self._x: self._x_train, self._y: self._y_train}
        for i in range(n_steps):
            _,loss = self._session.run([self._train,self._l],feed_dict=feed)
        
        self._fit = True
        #TODO print training report

    def predict(self, x_test, 
                predict_classes=False,
                batches=False,
                shuffle_batches=True,
                decimals=0):
        
        #dimension handling
        if x_test.ndim == 1:
            x_test = x_test.reshape(x_test.shape[0],1)
        
        y_out = self._session.run(self._y_pred,feed_dict={self._x:x_test})
        if predict_classes:
            y_out = np.round(y_out, decimals=decimals)
        return y_out
            
    def report(self):
        #get tensorboard information etc
        pass
    
    def save(self):
        #save model
        pass
    
    def start_session(self,**args):
        self._session = tf.Session(**args)
        
    def close_session(self, **args):
        self._session.close(**args)