import tensorflow as tf

class Agent(object):
    def __init__(self):
        ################################################################
        # Action probability
        ################################################################
        with tf.variable_scope("ActionProb"):
            self.chosen_strokes = tf.placeholder(tf.float32, [1, None, 256 + 10], name="ph_chosen_strokes")
            self.num_chosen_strokes = tf.placeholder(tf.int32, [None], name="ph_num_chosen_strokes")
            self.rnn_dropout = tf.placeholder_with_default(tf.constant(1, dtype=tf.float32), None, name="ph_rnn_dropout")

            self.candidate_strokes = tf.placeholder(tf.float32, [None, 256 + 10], name="ph_candidate_strokes")
            self.num_candidate_strokes = tf.placeholder(tf.int32, [], name="ph_num_candidates")
            self.sketch_class = tf.placeholder(tf.float32, [None, 9], name="ph_sketch_class")
            self.full_dropout = tf.placeholder_with_default(tf.constant(1, dtype=tf.float32), None, name="ph_full_dropout")

            rnn_cell = tf.contrib.rnn.GRUCell(64)
            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, input_keep_prob=self.rnn_dropout, output_keep_prob=self.rnn_dropout, state_keep_prob=self.rnn_dropout)
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(rnn_cell, self.chosen_strokes, sequence_length=self.num_chosen_strokes,dtype=tf.float32)

            chosen_strokes_embedding = tf.tile(tf.contrib.layers.fully_connected(rnn_outputs[:,-1], 18, activation_fn=tf.nn.tanh), [self.num_candidate_strokes, 1])
            candidate_strokes_embedding = tf.contrib.layers.fully_connected(self.candidate_strokes, 9, activation_fn=tf.nn.tanh)
            sketch_class_embedding = tf.contrib.layers.fully_connected(self.sketch_class, 3, activation_fn=tf.nn.tanh)

            full_embedding = tf.concat([chosen_strokes_embedding, candidate_strokes_embedding, sketch_class_embedding], axis=1)
            fc_layer = tf.contrib.layers.dropout(tf.contrib.layers.fully_connected(full_embedding, 15, activation_fn=tf.nn.tanh), keep_prob=self.full_dropout)

            self.logits = tf.transpose(tf.contrib.layers.fully_connected(fc_layer, 1, activation_fn=None))
            self.log_prob = tf.squeeze(tf.nn.log_softmax(self.logits), axis=0)

            self.action = tf.multinomial(self.logits, 1)

        ################################################################
        # Policy learning
        ################################################################
        with tf.variable_scope("PolicyLearning"):
            self.loss = -tf.gather(self.log_prob, self.action[0,0])

            tvars = [var for var in tf.trainable_variables() if "ActionProb" in var.name]
            self.gradient_holders = [tf.placeholder(tf.float32,name='%d_holder'%i) for i in range(len(tvars))]
            self.gradients = tf.gradients(self.loss, tvars)

            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.update_grad = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
