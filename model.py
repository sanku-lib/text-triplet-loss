import tensorflow as tf
import fasttext
import pickle
import numpy as np


class TripletLoss:

    def __init__(self, sequence_length, vocab_size, embedding_size, filter_sizes, num_filters,output_embedding_size, dropout_keep_prob,embeddings_lookup, l2_reg_lambda=0.0):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.output_embedding_size = output_embedding_size
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_loss = tf.constant(0.0)
        self.embeddings_lookup = embeddings_lookup

    def conv_net(self, input_x, reuse=False):
        
        # Embedding layer
        with tf.variable_scope("embedding", reuse=reuse):
            self.W = tf.Variable(self.embeddings_lookup,shape=[self.vocab_size, self.embedding_size], name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total], name="pool_flat")

        # Pre final layer / Embeddings Layer
        with tf.variable_scope("output", reuse=reuse):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.output_embedding_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[self.output_embedding_size]), name="b2")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b2)
            out_net = tf.nn.xw_plus_b(self.h_pool_flat, W, b2, name="scores")

        return out_net

    def triplet_loss(self, model_anchor, model_positive, model_negative, margin):
        distance1 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_positive, 2), 1, keepdims=True))
        distance2 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_negative, 2), 1, keepdims=True))
        return tf.reduce_mean(tf.maximum(distance1 - distance2 + margin, 0)) + 1e-9
