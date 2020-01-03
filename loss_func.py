from __future__ import print_function
import tensorflow as tf


def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\ sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    epsilon = 1e-14
    A = tf.linalg.matmul(inputs, true_w, transpose_b = True) # (batch_size x embedding_size) x (embedding_size x batch_size)
    B = tf.reduce_sum(tf.exp(A), axis = 1) # (batch_size x 1)
    B = tf.log(B + epsilon) # (batch_size x 1)
    A = tf.diag_part(A) # (batch_size x 1)
    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weights: Weights for nce loss. Dimension is [Vocabulary, embedding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here


    ==========================================================================
    # First, we need to get the weights for the corresponding labels from the weight matrix
    """
    embedding_size = tf.shape(inputs)[0]
    k = len(sample)
    epsilon = 1e-14

    labels = tf.reshape(labels,[labels.get_shape()[0],])
    sample = tf.reshape(sample,[k,])

    u_c = inputs

    u_o = tf.nn.embedding_lookup(weights, labels)           # batch_size x embedding_size
    # Also get the corresponding biases for outer words
    b_o = tf.nn.embedding_lookup(biases, labels)            # batch_size x 1
    # Get the unigram probabilities for outer words
    probs_o = tf.gather(unigram_prob, labels)  # batch_size x 1
    
    s_o = tf.diag_part(tf.matmul(u_c, u_o, transpose_b=True)) + b_o # (batch_size x embedding_size) x (embedding_size x batch_size)
    sigmoid_o = tf.sigmoid(s_o - tf.log(tf.scalar_mul(k, probs_o) + epsilon)) # batch_size x batch_size
    j_first_term = tf.scalar_mul(-1, tf.log(sigmoid_o + epsilon)) # batch_size x batch_size
    
    u_x = tf.nn.embedding_lookup(weights, sample)           # num_sampled x embedding_size
    b_x = tf.nn.embedding_lookup(biases, sample)            # num_sampled x 1
    probs_x = tf.gather(unigram_prob, sample)  # num_sampled x 1

    s_x = tf.matmul(u_c, u_x, transpose_b=True) + tf.transpose(b_x) # (batch_size x embedding_size) x (embedding_size x batch_size)
    sigmoid_x = tf.sigmoid(s_x - tf.transpose(tf.log(tf.scalar_mul(k, probs_x) + epsilon))) # batch_size x batch_size
    j_second_term = tf.scalar_mul(-1, tf.reduce_sum(tf.log(1-sigmoid_x+epsilon), axis=1)) # batch_size x batch_size

    j_first_term = tf.reshape(j_first_term,[embedding_size,1])
    j_second_term = tf.reshape(j_second_term,[embedding_size,1])
    return j_first_term + j_second_term