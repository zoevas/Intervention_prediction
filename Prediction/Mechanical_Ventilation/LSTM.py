import functools
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1  import set_random_seed
from tensorflow.python.framework import ops
from tensorflow.python.client import device_lib
print('device_lib.list_local_devices()',device_lib.list_local_devices())



BATCH_SIZE = 128
EPOCHS = 12
KEEP_PROB = 0.8
REGULARIZATION = 0.001
NUM_HIDDEN = [512, 512]
RANDOM = 0
NUM_CLASSES = 4

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class VariableSequenceLabelling:

    def __init__(self, data, target, dropout_prob, reg, num_hidden, class_weights):
        self.data = data
        self.target = target
        self.dropout_prob = dropout_prob
        self.reg = reg
        self._num_hidden = num_hidden
        self._num_layers = len(num_hidden)
        self.num_classes = len(class_weights)
        self.attn_length = 0
        self.class_weights = class_weights
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def make_rnn_cell(self,
                      attn_length=0,
                      base_cell=tf.compat.v1.nn.rnn_cell.BasicRNNCell):

        attn_length = self.attn_length
        input_dropout = self.dropout_prob
        output_dropout = self.dropout_prob

        cells = []
        for num_units in self._num_hidden:
            cell = base_cell(num_units)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout,
                                                seed=RANDOM)
            cells.append(cell)

        cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        return cell


    # predictor for slices
    @lazy_property
    def prediction(self):

        cell = self.make_rnn_cell

        # Recurrent network.
        output, final_state = tf.nn.dynamic_rnn(cell,
            self.data,
            dtype=tf.float32
        )

        with tf.variable_scope("model") as scope:
            tf.get_variable_scope().reuse_variables()

            # final weights
            num_classes = self.num_classes
            weight, bias = self._weight_and_bias(self._num_hidden[-1], num_classes)

            # flatten + sigmoid
            if self.attn_length > 0:
                logits = tf.matmul(final_state[0][-1][-1], weight) + bias
            else:
                logits = tf.matmul(final_state[-1][-1], weight) + bias

            prediction = tf.nn.softmax(logits)

            return logits, prediction


    @lazy_property
    def cross_ent(self):
        predictions = self.prediction[0]
        real = tf.cast(tf.squeeze(self.target), tf.int32)
        weights = tf.gather(self.class_weights, real)

        xent = tf.losses.sparse_softmax_cross_entropy(labels=real, logits=predictions, weights=weights)
        loss = tf.reduce_mean(xent) #shape 1
        ce = loss
        l2 = self.reg * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        ce += l2
        return ce

    @lazy_property
    def optimize(self):
        learning_rate = 0.0003
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.cross_ent)

    @lazy_property
    def error(self):
        prediction = tf.argmax(self.prediction[1], 1)
        real = tf.cast(self.target, tf.int32)
        prediction = tf.cast(prediction, tf.int32)
        mistakes = tf.not_equal(real, prediction)
        mistakes = tf.cast(mistakes, tf.float32)
        mistakes = tf.reduce_sum(mistakes, reduction_indices=0)
        total = 128
        mistakes = tf.divide(mistakes, tf.to_float(total))
        return mistakes

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


    @lazy_property
    def summaries(self):
        tf.summary.scalar('loss', tf.reduce_mean(self.cross_ent))
        tf.summary.scalar('error', self.error)
        merged = tf.summary.merge_all()
        return merged

def LSTM_Model(x_train, y_train, x_test, x_val):
    class_weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    print('class_weight', class_weight)
    classes = np.unique(y_train)
    print('classes', classes)
    print('LSTM starting')
    ops.reset_default_graph()
    set_random_seed(RANDOM)
    config = tf.compat.v1.ConfigProto(allow_soft_placement = True)

    sess = tf.compat.v1.Session(config = config)
    with  sess, tf.device('/cpu:0'):
        _, length, num_features = x_train.shape
        num_data_cols = num_features

        data = tf.placeholder(tf.float32, [None, length, num_data_cols])
        target =  tf.placeholder(tf.float32, [None])
        dropout_prob =  tf.placeholder(tf.float32)
        reg =  tf.placeholder(tf.float32)



        # initialization
        model = VariableSequenceLabelling(data, target, dropout_prob, reg, num_hidden=NUM_HIDDEN, class_weights=class_weight)
        sess.run(tf.global_variables_initializer())

        batch_size = BATCH_SIZE
        dp = KEEP_PROB
        rp = REGULARIZATION
        train_samples = x_train.shape[0]
        indices = list(range(train_samples))
        num_classes = NUM_CLASSES

        # for storing results
        test_data = x_test
        val_data = x_val

        val_aucs = []
        test_aucs = []
        val_aucs_macro = []
        test_aucs_macro = []
        test_accuracys = []
        test_f1s = []
        test_auprcs = []

    epoch = -1
