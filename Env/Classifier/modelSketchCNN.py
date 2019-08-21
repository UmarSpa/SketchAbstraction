import tensorflow as tf
import tensorflow.contrib.slim as slim

class Classifier():
    def __init__(self):

        ####################################################################
        # Placeholders for the network
        ####################################################################
        with tf.variable_scope("Input"):
            self.input_data = tf.placeholder(shape=[None, 225, 225, 1], dtype=tf.float32, name="X")
            self.labels = tf.placeholder(shape=[None], dtype=tf.int32, name="Y")

            self.dropoutFlag = tf.placeholder(shape=(),dtype=tf.bool, name="dropoutFlag")
        ####################################################################
        # Outputs and state computation
        ####################################################################
        with tf.variable_scope("Main"):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                with slim.arg_scope([slim.conv2d], padding='VALID'):
                    conv1 = slim.conv2d(self.input_data, 64, [15, 15], 3, scope='conv1_s1')
                    pool1 = slim.max_pool2d(conv1, [3, 3], scope='pool1')
                    conv2 = slim.conv2d(pool1, 128, [5, 5], scope='conv2_s1')
                    pool2 = slim.max_pool2d(conv2, [3, 3], scope='pool2')
                    conv3 = slim.conv2d(pool2, 256, [3, 3], padding='SAME', scope='conv3_s1')
                    conv4 = slim.conv2d(conv3, 256, [3, 3], padding='SAME', scope='conv4_s1')
                    conv5 = slim.conv2d(conv4, 256, [3, 3], padding='SAME', scope='conv5_s1')
                    pool5 = slim.max_pool2d(conv5, [3, 3], scope='pool5')
                    pool5 = slim.flatten(pool5)
                    #conv5 = tf.reshape(conv5, [-1,7*7*64])
                    fc6 = slim.fully_connected(pool5, 512, scope='fc6_s1')
                    dropout6 = slim.dropout(fc6, 0.5, is_training=self.dropoutFlag, scope='dropout6')
                    fc7 = slim.fully_connected(dropout6, 512, scope='fc7_s1')
                    dropout7 = slim.dropout(fc6, 0.5, is_training=self.dropoutFlag, scope='dropout7')
                    fc8 = slim.fully_connected(dropout7, 9, activation_fn=None, scope='fc8_sketchANet')
                    self.fc8_preds = tf.nn.softmax(fc8)
                    self.finalLabel = tf.argmax(self.fc8_preds, 1)
                    self.values, self.indices = tf.nn.top_k(self.fc8_preds, k=9)

        ####################################################################
        # Classification - Loss
        ####################################################################
        with tf.variable_scope("Loss"):
            lossB = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=fc8)
            loss = tf.reduce_mean(lossB)
            self.loss = loss

        ####################################################################
        # Classification - Accuracy
        ####################################################################
        with tf.variable_scope("Accuracy"):
            correct = tf.equal(tf.cast(tf.argmax(self.fc8_preds,1), tf.int32), self.labels)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            self.accuracy = accuracy

        ####################################################################
        # Optimizer
        ####################################################################
        with tf.variable_scope("Optimizer"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        ####################################################################
        # Summaries
        ####################################################################
        with tf.variable_scope("Summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            self.summary_op = tf.summary.merge_all()