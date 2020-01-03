import os
import pickle
import numpy as np

from scipy import spatial

model_path = './models/'
loss_model = 'nce'
# loss_model = 'cross_entropy'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply   

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

dataset_type = 'test' # dev OR test

input_file_name = 'word_analogy_{}.txt'.format(dataset_type)
output_file_name = 'word_analogy_{}_predictions_{}.txt'.format(dataset_type, loss_model)

def find_analogies():
    with open(input_file_name,"r") as input_file, open(output_file_name,"w+") as output_file:
        result = ""
        for line in input_file:
            line.strip()
            examples, choices = line.split("||")
            examples_word_pairs = examples.strip().split(",")
            choices_word_pairs = choices.strip().split(",")
            cosine_score_list = []
            choice_examples_cosine_sim = []
            example_vector_differences = []
            # First we will loop over the example pairs and calculate their mean cosine vector
            # This mean will capture their relation
            for word_pair in examples_word_pairs:
                word1, word2 = word_pair.strip('"').split(":")
                word_id1, word_id2 = dictionary[word1],dictionary[word2]
                embed1, embed2 = embeddings[word_id1],embeddings[word_id2]
                cosine_score = embed2 - embed1
                example_vector_differences.append(cosine_score)

            # Calculate the mean cosine 
            avg_vector = np.mean(example_vector_differences, axis = 0)

            # Now we will find the cosine score for choices/prediction pairs
            for word_pair in choices_word_pairs:
                word1, word2 = word_pair.strip('"').split(":")
                word_id1, word_id2 = dictionary[word1],dictionary[word2]
                embed1, embed2 = embeddings[word_id1],embeddings[word_id2]
                vector = embed2 - embed1
                choice_examples_cosine_sim.append(1 - spatial.distance.cosine(avg_vector, vector))

            max_index = choice_examples_cosine_sim.index(max(choice_examples_cosine_sim))
            min_index = choice_examples_cosine_sim.index(min(choice_examples_cosine_sim))
            result += choices.strip().replace(",", " ") + " " + choices_word_pairs[max_index].strip() + " " + \
                      choices_word_pairs[min_index] + "\n"

        output_file.write(result)


def find_top20_similar():
    input_words = ['first', 'american', 'would']
    top_20_similarity = {word : [] for word in input_words}
    for word, word_id in dictionary.items():
        word_vector1 = embeddings[word_id]
        # calculate the cosine similarity of this word with 'first', 'american' and 'would'
        for ip_word, top_20 in top_20_similarity.items():
            word_vector2 = embeddings[dictionary[ip_word]]
            top_20.append((1 - spatial.distance.cosine(word_vector1, word_vector2),  word))
       
    for word, similarity in top_20_similarity.items():
        print(word)
        print(sorted(top_20_similarity[word], key=lambda tup: tup[0], reverse=True)[:22])
 
if __name__ == '__main__':
    find_analogies()
    # find_top20_similar()