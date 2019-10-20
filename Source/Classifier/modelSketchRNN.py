import tensorflow as tf

class Classifier(object):
    def __init__(self):

        ####################################################################
        # Placeholders for the network
        ####################################################################
        with tf.variable_scope("Placeholders"):
            self.input_data = tf.placeholder(tf.float32, [None, None, 3], name="InputData")
            self.input_data_lens = tf.placeholder(tf.int32, [None], name="InputLen")
            self.labels = tf.placeholder(tf.int32, [None], name="InputLabel")
            self.input_dropout = tf.placeholder_with_default(tf.constant(1, dtype=tf.float32), None, name="in_rnn_dropout")
            self.output_dropout = tf.placeholder_with_default(tf.constant(1, dtype=tf.float32), None,name="out_rnn_dropout")

        ####################################################################
        # Outputs and state computation
        ####################################################################
        with tf.variable_scope("Main"):
            fw_cells = []
            temp_fw_cell = tf.contrib.rnn.BasicLSTMCell(256)
            temp_fw_cell = tf.contrib.rnn.DropoutWrapper(temp_fw_cell, input_keep_prob=self.input_dropout)
            fw_cells.append(temp_fw_cell)
            rnn_cell = tf.contrib.rnn.MultiRNNCell(fw_cells)
            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=self.output_dropout)
            outputs, stateFinal = tf.nn.dynamic_rnn(rnn_cell, self.input_data, sequence_length=self.input_data_lens, dtype=tf.float32)
            self.last_output = stateFinal[-1].h

        ####################################################################
        # Classification Branch
        ####################################################################
        with tf.variable_scope("Classification"):
            classifier_W = tf.Variable(tf.truncated_normal([256, 9]))
            classifier_b = tf.Variable(tf.constant(0.1, shape=[9]))
            self.logits = tf.nn.xw_plus_b(self.last_output, classifier_W, classifier_b)
            self.preds = tf.nn.softmax(self.logits)
            self.values, self.indices = tf.nn.top_k(self.preds, k=9)

        ####################################################################
        # Classification - Loss
        ####################################################################
        with tf.variable_scope("Loss"):
            lossB = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = tf.reduce_mean(lossB)
            self.loss = loss

        ####################################################################
        # Classification - Accuracy
        ####################################################################
        with tf.variable_scope("Accuracy"):
            correct = tf.equal(tf.cast(tf.argmax(self.preds, 1), tf.int32), self.labels)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            self.accuracy = accuracy
