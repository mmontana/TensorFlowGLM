import tensorflow as tf

class GLMBase(object):
    
    def __init__(self,x_train,y_train,activation=None,loss=tf.losses.mean_squared_error,
                 optim=tf.train.AdamOptimizer(.1), dropout= None):
            
        self._fit = False
        self._session = None
        self._x_train = x_train
        self._y_train = y_train
        #get all class variables
        
        self._activation = activation
        
        self._loss = loss
        self._optim = optim
        
        #TODO generalize dimensionality
        self._W_init = tf.truncated_normal(shape=[1])
        self._b_init = tf.truncated_normal(shape=[1])
        
        self._W = tf.Variable(self._W_init)
        self._b = tf.Variable(self._b_init)
        
        self._x = tf.placeholder(dtype=tf.float32)
        self._y = tf.placeholder(dtype=tf.float32)
        self._y_lin = (self._W * self._x) + self._b
        
        if self._activation is not None:
            self._y_pred = self._activation(self._y_lin) 
        else:
            self._y_pred = self._y_lin_pred
            
        self._l = self._loss(self._y, self._y_pred)
        
        self._train = self._optim.minimize(self._l)
    
    def fit(self, n_steps):
        #create session
        if not self._fit:
            self._session = tf.Session()
            self._session.run(tf.global_variables_initializer())
        
        #batch training goes here
        
        feed = {self._x: self._x_train, self._y: self._y_train}
        
        #gradient descent
        for i in range(n_steps):
            _,loss = self._session.run([self._train,self._l],feed_dict=feed)
        
        #store weights?           
        pass
    
    #TODO predict vs predict proba for logistic cases
    def predict(self,x_test):
        return self._session.run(self._y_pred,feed_dict={self._x:x_test})    
    
    def fit_predict(self):
        pass
        #wrapper for self.fit() self.predict()
            
    def report(self):
        #get tensorboard information etc
        pass
    
    def save(self):
        #save model
        pass