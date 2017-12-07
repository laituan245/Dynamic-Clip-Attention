import tensorflow as tf

K_MAX_STRATEGY = 0
K_THRESHOLD_STRATEGY = 1

def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.1))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='VALID')
    return tf.nn.relu(conv + biases)

class Model:
    def __init__(self, params):
        # Extract required parameters
        batch_size = params['batch_size']
        max_sent_len = params['max_sent_len']
        num_labels = params['num_labels']
        vocab_size = params['vocab_size']
        embedding_dim = params['embedding_dim']
        clipping_strategy = params['clipping_strategy']
        clipping_k_1 = params['clipping_k_1']
        clipping_k_2 = params['clipping_k_2']
        out_channel_size = params['out_channel_size']
        hidden_layer_size = params['hidden_layer_size']
        learning_rate = params['learning_rate']

        with tf.name_scope('placeholders'):
            self.keep_prob = tf.placeholder('float32', shape=(), name='drop')
            self.sent1 = tf.placeholder(tf.int32, shape=(batch_size, max_sent_len))
            self.sent1_length = tf.placeholder(tf.int32, shape=(batch_size))
            self.sent2 = tf.placeholder(tf.int32, shape=(batch_size, max_sent_len))
            self.sent2_length = tf.placeholder(tf.int32, shape=(batch_size))
            self.target = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        with tf.variable_scope('embedding'):
            embedding = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False)
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
            self.embedding_init = embedding.assign(self.embedding_placeholder)

            sent1_emb = tf.nn.embedding_lookup(embedding, self.sent1)
            sent2_emb = tf.nn.embedding_lookup(embedding, self.sent2)

        with tf.variable_scope('preprocessing'):
            v1 = tf.get_variable('v1', [embedding_dim, hidden_layer_size])
            b1 = tf.Variable(tf.zeros([hidden_layer_size]), name='bias_b1')
            v2 = tf.get_variable('v2', [embedding_dim, hidden_layer_size])
            b2 = tf.Variable(tf.zeros(hidden_layer_size), name='bias_b2')

            tiled_v1 = tf.tile(tf.expand_dims(v1, axis=0), [batch_size, 1, 1])
            tiled_b1 = tf.tile(tf.expand_dims(b1, axis=0), [max_sent_len, 1])
            tiled_v2 = tf.tile(tf.expand_dims(v2, axis=0), [batch_size, 1, 1])
            tiled_b2 = tf.tile(tf.expand_dims(b2, axis=0), [max_sent_len, 1])

            sent1_emb = tf.multiply(
                            tf.nn.sigmoid(tf.matmul(sent1_emb, tiled_v1) + tiled_b1),
                            tf.nn.tanh(tf.matmul(sent1_emb, tiled_v2) + tiled_b2)
            )

            sent2_emb = tf.multiply(
                            tf.nn.sigmoid(tf.matmul(sent2_emb, tiled_v1) + tiled_b1),
                            tf.nn.tanh(tf.matmul(sent2_emb, tiled_v2) + tiled_b2)
            )

        with tf.name_scope('alignment_and_comparision'):
            sent1_alignments = self._calculate_alignment(sent1_emb, self.sent1_length,
                                                         sent2_emb, self.sent2_length,
                                                         max_sent_len, clipping_strategy,
                                                         clipping_k_1)
            sent2_alignments = self._calculate_alignment(sent2_emb, self.sent2_length,
                                                         sent1_emb, self.sent1_length,
                                                         max_sent_len, clipping_strategy,
                                                         clipping_k_2)

            comparision_1 = tf.multiply(sent1_emb, sent1_alignments)
            comparision_1 = tf.expand_dims(comparision_1, -1)

            comparision_2 = tf.multiply(sent2_emb, sent2_alignments)
            comparision_2 = tf.expand_dims(comparision_2, -1)

        with tf.variable_scope('aggregation'):
            with tf.variable_scope("conv") as scope:
                sent1_vector = self._extract_features_cnn(comparision_1, hidden_layer_size, out_channel_size)

            with tf.variable_scope("conv", reuse=True):
                sent2_vector = self._extract_features_cnn(comparision_2, hidden_layer_size, out_channel_size)

            final_vector = tf.concat([sent1_vector, sent2_vector], axis=1)

        with tf.variable_scope('output_layer'):
            v = tf.get_variable('final_v', [2 * out_channel_size, num_labels])
            b = tf.Variable(tf.zeros([num_labels]), name='final_bias_b')

            self.scores = tf.matmul(final_vector, v) + b
            self.prob = tf.nn.softmax(self.scores)
            self.prediction = tf.argmax(self.scores, 1)

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.target, logits = self.scores)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_preds = tf.equal(tf.argmax(self.target, 1), self.prediction)
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
            self.correct = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

        with tf.name_scope("optimization"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gvs = optimizer.compute_gradients(self.loss)
            self.optimize = optimizer.apply_gradients(gvs)

        # Operation for initializing the variables
        self.init = tf.global_variables_initializer()

        # Operation to save and restore all the variables.
        self.saver = tf.train.Saver()

    def _extract_features_cnn(self, comparision_vector, hidden_size, out_channel_size):
        with tf.variable_scope('trigram'):
            features = conv_relu(comparision_vector, [3, hidden_size, 1, out_channel_size], [out_channel_size])
            features = tf.squeeze(features)
            features = tf.reduce_max(features, axis=1) # Max-pooling
        return features

    def _calculate_alignment(self, sent1_emb, sent1_length, sent2_emb, sent2_length,
                             max_sent_len, clipping_strategy, clipping_k):
        sent1_mask = tf.sequence_mask(sent1_length, max_sent_len, tf.float32)
        sent2_mask = tf.sequence_mask(sent2_length, max_sent_len, tf.float32)
        similarity_matrix = tf.matmul(sent1_emb, tf.transpose(sent2_emb, [0, 2 ,1]))
        weight_matrix = tf.nn.softmax(similarity_matrix)
        weight_matrix = tf.multiply(weight_matrix, tf.expand_dims(sent2_mask, dim=1))
        weight_matrix = tf.divide(weight_matrix, tf.reduce_sum(weight_matrix, axis=2, keep_dims=True))

        # Dynamic-Clip Attention
        if clipping_strategy == K_MAX_STRATEGY:
            top_k_values, top_k_indices = tf.nn.top_k(weight_matrix, clipping_k)
            threshold = tf.reduce_min(top_k_values, axis=2, keep_dims=True)
            boolean_matrix = tf.greater_equal(weight_matrix, threshold)
            weight_matrix = tf.multiply(weight_matrix, tf.to_float(boolean_matrix))
        elif clipping_strategy == K_THRESHOLD_STRATEGY:
            boolean_matrix = tf.greater_equal(weight_matrix, clipping_k)
            weight_matrix = tf.multiply(weight_matrix, tf.to_float(boolean_matrix))
        weight_matrix = tf.divide(weight_matrix, tf.reduce_sum(weight_matrix, axis=2, keep_dims=True))

        sent1_alignments = tf.matmul(weight_matrix, sent2_emb)
        return sent1_alignments

    def train(self, sess, batch, keep_prob = 0.5):
        _, sent1, sent1_length, sent2, sent2_length, labels = batch
        _, batch_loss, batch_accuracy = sess.run(
                        [self.optimize, self.loss, self.accuracy],
                        feed_dict = {
                            self.keep_prob: keep_prob,
                            self.sent1: sent1,
                            self.sent1_length: sent1_length,
                            self.sent2: sent2,
                            self.sent2_length: sent2_length,
                            self.target: labels,
                        })
        return batch_loss, batch_accuracy

    def predict(self, sess, batch):
        _, sent1, sent1_length, sent2, sent2_length, labels = batch
        batch_probability = sess.run(
                            self.prob,
                            feed_dict = {
                                self.keep_prob: 1,
                                self.sent1: sent1,
                                self.sent1_length: sent1_length,
                                self.sent2: sent2,
                                self.sent2_length: sent2_length,
                                self.target: labels,
                            })
        return batch_probability
