## Overview

In this project, I have implemented the Skip gram model using cross-entropy loss and NCE loss for learning word vectors. I have also performed a few experiments to tune different hyperparameters and compare the two loss functions. After learning the word vectors, I have used them in a word analogy task as described in the section below.

### Best Models

#### NCE Loss

* Accuracy - 35.7%

Configuration:
* batch_size - 256
* skip_window - 4
* num_skips - 8
* embedding_size - 128
* max_num_steps - 200001
* learning rate - 5


#### Cross Entropy
* Accuracy - 33.9%
* loss_value - 4.8267

Configuration:
* batch_size - 256
* skip_window - 4
* num_skips - 8
* embedding_size - 128
* max_num_steps - 200001
* learning rate - 1.0


I have implemented the following functions:

* `generate_batch()` for skip-gram model in `word2vec_basic.py`
* Cross Entropy loss `cross_entropy_loss()` in `loss_func.py`
* NCE loss `nce_loss()` in `loss_func.py`
* `find_analogies()` in `word_analogy.py`
* `find_top20_similar()` in `word_analogy.py`

#### Generating batches for skip-gram model
* The function `generate_batch()` is called by the `train()` method to genarate a mini-batch of inputs and labels in every epoch.
* The inputs to this function are - 
	* `data` - the entire vocabulary
	* `batch_size` - the number of data points in one batch
	* `num_skips` - the number of samples to draw in a window
	* `skip_window` - how many words to consider left and right from a context word.
* We use a global variable called `data_index` to maintain the positioning in the entire corpus whenever this function is called. Not updating this will be equivalent to generating the same batch again and again for every epoch.
* Created `window_size = skip_window*2 + 1`
* We first calculate the position of our centre word in the current batch by `skip_window + data_index`.
* Then we sample `num_skips` words from this window by slicing the data w.r.t. to the centre word.

#### loss_func.py - function called during training 
1. cross_entropy_loss(inputs, true_w)

* Inputs:
	* inputs: Embeddings for context words with dimension [batch_size, embedding_size].
    * true_w: Embeddings for predicting words with dimension [batch_size, embedding_size].
* Calculating `A = log(exp({u_o}^T v_c))`
	* I multiplied inputs with transpose of true_w. Then I extracted the diagnol elements using tf.diag_part.
* Calculating `B = log(\ sum{exp({u_w}^T v_c)})`
	* I re-used the multiplication of inputs with transpose of true_w from previous calculations and then applied exponent function. Then I took a sum across all the columns.
	* Before taking the log, I added a very small value 1e-14 to the input of log to avoid the problem of NaN.
* The final value of the loss is B - A.


2. nce_loss(inputs, weights, biases, labels, sample, unigram_prob)

* Inputs: 
	* inputs - Embeddings for context words with dimension [batch_size, embedding_size]
	* weights - Weights for nce loss with dimension [Vocabulary, embedding_size]
	* biases - Biases for nce loss with dimension [Vocabulary, 1]. 
	* labels: Word_ids for predicting words with dimension [batch_size, 1],
	* samples: Word_ids for negative samples with dimension [num_sampled]
	* unigram_prob: Unigram probabilitity with dimension  [Vocabulary].
* I first converted the Python lists - labels, sample  to tensors using and then extracted the relevant weights, biases for labels and sample using `tf.nn.embedding_lookup()`.
* Then I constructed the two different parts of the nce loss separately one by one.
* The first term involves multiplying the inputs and labels, multiplying unigram probabilities with k (the number of samples) and adding a very small value epsilon before applying the log.
* We apply the sigmoid to the multiplication of inputs and labels and then subtract the previous calculation from this term.
* We do the similar steps for calculating the second term of NCE loss.
* The final value of NCE loss is the addition of both the terms. Please note that I have multiplied these two terms with a  scalar "-1" already so we need not return a negative value here.


#### Word Analogy - `word_analogy.py`
* `find_analogies()` - this function evaluates the analogy between 'example' pairs and 'choices' pairs and find the most and least representative pairs from 'choice' pairs.
* I have created this file such that it can be run on both - dev and test dataset by changing the `dataset_type` variable.
* It can also use either NCE model or Cross Entropy model by configuring the `loss_model` variable.
* The `input_file_name` points to the file containing the dataset.
* In `find_analogies()`, we basically find the word embeddings or vectors for the 'example' word pairs and then model the relationship in every pair by taking a difference of the word vectors. Let us call this relationship vector.
* We then take the mean of all the relationship vectors in the example words.
* Then we iterate over all the 'choice' pairs and find a vector for all the word pairs. Then we find the cosine similarity between these vectors and the mean relationship vector that we found in the previous step.
* We then return the word pairs with highest and lowest cosine similarity with the relationship vector.
* `find_top20_similar()` - this function finds the top 20 most similar words to the given words - first, american and would.
* We find the similar words by finding the cosine similarity between the given words and all the words in the dictionary.
* For each given word, we print the top 20 most similar words.