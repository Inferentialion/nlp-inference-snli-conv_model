from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
import os
import random

# Get the current working directory for the pickles:
current_directory = os.getcwd()


def get_vocab(dataset):
    """function specific and self-sufficient for getting the vocabulary out of a dataset generator"""
    tokenizer = tfds.features.text.Tokenizer()
    vocab = set()

    for element in dataset:
        # only tokenize the str elements of the dataset:
        hypothesis = element['hypothesis']
        premise = element['premise']
        hypothesis_tokenized = tuple(tokenizer.tokenize(hypothesis))
        premise_tokenized = tuple(tokenizer.tokenize(premise))

        joint_tokens = hypothesis_tokenized + premise_tokenized
        for token in joint_tokens:
            vocab.add(token)

    # put it into a pickle:
    with open(f'{current_directory}/generator_vocab.pickle', 'wb') as pickle_file:
        pickle.dump(vocab, pickle_file)

    return vocab


def gen_to_encoded_list(generator_dataset, vocab, split='train'):

    encoder = tfds.features.text.TokenTextEncoder(vocab)

    sentence_pairs_dict = {'hypotheses': [], 'premises': [], 'labels': []}

    for element in generator_dataset:
        if element['label'] >= 0:  # filter -1 labeled elements (i.e. without clear label when asked the participants)
            # Encode tokenized sentences to integers:
            encoded_hypothesis = encoder.encode(element['hypothesis'])
            encoded_premise = encoder.encode(element['premise'])
            # Append each encoded element to its corresponding dict key:
            sentence_pairs_dict['hypotheses'].append(encoded_hypothesis)
            sentence_pairs_dict['premises'].append(encoded_premise)
            sentence_pairs_dict['labels'].append(element['label'])

    assert len(sentence_pairs_dict['hypotheses']) == len(sentence_pairs_dict['premises']) \
           == len(sentence_pairs_dict['labels'])

    # Write a pickle:
    if split == 'train':
        with open(f'{current_directory}/encoded_list_train.pickle', 'wb') as pickle_file:
            pickle.dump(sentence_pairs_dict, pickle_file)

        return sentence_pairs_dict

    elif split == 'validation':
        with open(f'{current_directory}/encoded_list_val.pickle', 'wb') as pickle_file:
            pickle.dump(sentence_pairs_dict, pickle_file)

        return sentence_pairs_dict
    else:
        with open(f'{current_directory}/encoded_list_test.pickle', 'wb') as pickle_file:
            pickle.dump(sentence_pairs_dict, pickle_file)

        return sentence_pairs_dict


def encoded_list_to_dataset(sentence_pairs_dict, batch_size, padded_shapes):

    MAX_LEN = padded_shapes[0]

    sentence_pairs_dict['hypotheses'] = tf.keras.preprocessing.sequence.pad_sequences(sentence_pairs_dict['hypotheses'],
                                                                                      maxlen=MAX_LEN, padding='post')
    sentence_pairs_dict['premises'] = tf.keras.preprocessing.sequence.pad_sequences(sentence_pairs_dict['premises'],
                                                                                    maxlen=MAX_LEN, padding='post')

    # Quick sanity ckeck:
    num_elements = len(sentence_pairs_dict['hypotheses'])
    print("Printing num_elements in the hypotheses key: ", num_elements)

    dataset = tf.data.Dataset.from_tensor_slices(sentence_pairs_dict)
    dataset = dataset.shuffle(551000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def create_toy_dataset():
    toy_values, toy_labels = [], []
    for _ in range(500):
        list_length = random.randint(5, 80)
        toy_values.append(tf.random.uniform((list_length,)).numpy().tolist())

    toy_dataset = {'hypotheses': toy_values, 'premises': toy_values}

    return toy_dataset

