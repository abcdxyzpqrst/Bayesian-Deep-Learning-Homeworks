import numpy as np
import tensorflow as tf

np.random.seed(2019)
tf.set_random_seed(2019)

class MLP:
    def __init__(self, n_hidden=30, n_output=1):
        self.n_hidden = n_hidden
        self.n_output = n_output

        # define placeholders for training MLP
        self.x = tf.placeholder(tf.float32, shape=[None, 1])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.dropout_rate = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)

        self.predictions, self.loss, self.opt = self.build_graph(n_hidden=self.n_hidden, n_output=self.n_output)

    def build_graph(self, n_hidden=512, n_output=10):
        """
        Build computational graph for toy dataset with dropout

        Args:
            n_hidden: the number of hidden units in MLP.
            n_output: the size of output layer (=1)

        Returns:
            prob: probability of prediction (m, 1)
            loss: corss entropy loss
            learning rate: learning rate for optimizer (ex. SGD, RMSprop, Adam, etc.)
        """

        with tf.variable_scope('mlp'):
            # initializers for weight and bias
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.0)

            # 1st hidden layer (input dimension --> n_hidden)
            w0 = tf.get_variable('w0', [self.x.get_shape()[1], n_hidden], initializer=w_init)
            b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
            h0 = tf.matmul(self.x, w0) + b0
            h0 = tf.nn.relu(h0)
            h0 = tf.nn.dropout(h0, rate=self.dropout_rate)

            # 2nd hidden layer (n_hidden --> n_hidden)
            w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
            b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.relu(h1)
            h1 = tf.nn.dropout(h1, rate=self.dropout_rate)

            # 3nd hidden layer (n_hidden --> n_hidden)
            w2 = tf.get_variable('w2', [h1.get_shape()[1], n_hidden], initializer=w_init)
            b2 = tf.get_variable('b2', [n_hidden], initializer=b_init)
            h2 = tf.matmul(h1, w2) + b2
            h2 = tf.nn.relu(h2)
            h2 = tf.nn.dropout(h2, rate=self.dropout_rate)

            # output layer (n_hidden --> n_output)
            wo = tf.get_variable('wo', [h2.get_shape()[1], n_output], initializer=w_init)
            bo = tf.get_variable('bo', [n_output], initializer=b_init)
            predictions = tf.matmul(h2, wo) + bo
            
            # we don't need probability --> just regression problems
            #prob = tf.nn.softmax(logit, axis=1)
            # loss
            #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            #    logits=logit, labels=self.y))
            loss = tf.losses.mean_squared_error(
                labels=self.y,
                predictions=predictions
            )
            
            # optimizer
            opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        return predictions, loss, opt

mlp = MLP()
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())

"""
Toy dataset = sine function with standard Gaussian noise ~ N(0, 1^2)
Training data (1-dimensional) --> only from -2 to 2

Test data (1-dimensional) --> from -4 to 4
"""
x_train = np.linspace(-3, 3.2, 31, endpoint=False)
noise_train = 0.2 * np.random.randn(31)
y_train = np.sin(x_train) + noise_train

x_train, y_train = np.expand_dims(x_train, -1), np.expand_dims(y_train, -1)

x_test = np.linspace(-6, 6.2, 61, endpoint=False)
noise_test = np.random.randn(61)
y_test = np.sin(x_test) + 0.2 * noise_test

x_test, y_test = np.expand_dims(x_test, -1), np.expand_dims(y_test, -1)

from matplotlib import pyplot as plt
plt.figure()
plt.xlabel("$x$")
plt.ylabel("$y = \sin x + 0.2 \epsilon$ with $\epsilon$ ~ $N(0,1^2)$")
plt.plot(x_train, y_train, 'bo', color='black', label="Train data")
plt.plot(x_test, y_test, 'bo', color='red', label="Test data")
plt.legend(loc='best')
plt.savefig("data_statistics.png", bbox_inches="tight", dpi=300)
plt.close()

def train(net, sess, num_epoch=100):
    # iterating epoch
    for epoch in range(num_epoch):
        # here we use batch gradient descent since the data size is small!!
        feed_dict = {net.x: x_train,
                     net.y: y_train,
                     net.dropout_rate: 0.3,
                     net.lr: 1e-3}
        avg_loss, _ = sess.run([net.loss, net.opt], feed_dict=feed_dict)
        print ('Epoch: ', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_loss))
    print ("Learning finished")
    return

def evaluate(net, sess, T=30):
    repeat_predictions = []
    for i in range(T):
        repeat_predictions.append(sess.run(net.predictions, feed_dict={net.x: x_test,
                                                                       net.y: y_test,
                                                                       net.dropout_rate: 0.3}))
    """
    Important thing!!

    Here, repeat prediction has size of (T, 61, 1)
    T: How many runs for measuring uncertainty?
    61: the number of test data samples
    1: dimensions of outputs
    """
    return repeat_predictions

train(mlp, sess)
repeat_predictions = evaluate(mlp, sess)
repeat_predictions = np.array(repeat_predictions)

# mean, std calculations
mean = []
std = []

# MC dropout uncertainty figure
plt.figure()

