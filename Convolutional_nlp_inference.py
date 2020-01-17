import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
import time
import os
import datetime

# Imports with different functions to preprocess the dataset:
from snli_utils import get_vocab, gen_to_encoded_list, encoded_list_to_dataset

# Remember to set your own pickle directories as well as checkpoint and summary directories if you wish different ones.

# Get the dataset: -----------------------------------------------------------------------------------------------------
builder_obj = tfds.builder('snli')
print(builder_obj.info)

datasets_splits_dict = builder_obj.as_dataset()  # returns all splits in a dict
train_dataset, val_dataset, test_dataset = datasets_splits_dict["train"], datasets_splits_dict["validation"], \
                                           datasets_splits_dict["test"]

# Get the iterators to encode the elements into lists of integers representing the words for each sentence:
train_np, val_np, test_np = tfds.as_numpy(train_dataset), tfds.as_numpy(val_dataset), tfds.as_numpy(test_dataset)

# We first need the vocab ----------------------------------------------------------------------------------------------

if not os.path.isfile(r'./generator_vocab.pickle'):
    # Get the vocabulary for the first time:
    vocab = list(get_vocab([train_np, val_np, test_np]))
    print("\nVocab ready!")
else:
    # Get the vocab (obtained by 'get_vocab' function in snli_utils) from the pickle:
    with open(r'./generator_vocab.pickle', 'rb') as pickle_file:
        vocab = list(pickle.load(pickle_file))

print(len(vocab), vocab[:20])
print("\n")

# Encode the dataset to integers and then to tf.data.Dataset objects: --------------------------------------------------

# First set some useful hyperparams:
BATCH_SIZE = 500
MAX_LEN = 80
EMBEDDING_DIM = 300
padded_shapes = [MAX_LEN]

# encoded_list = create_toy_dataset()  # Optional smaller dataset to better try different configurations and debug
# our model.

# TRAIN: ----------------------------------------------------

if not os.path.isfile(r'./encoded_list_train.pickle'):
    # Get the encoded list of words for the train set for the first time:
    encoded_list_train = gen_to_encoded_list(train_np, vocab)
    print("\ntrain set encoded!")
else:
    # Load encoded list of words from pickle:
    with open(r'./encoded_list_train.pickle', 'rb') as pickle_file:
        encoded_list_train = pickle.load(pickle_file)

# Create tf.data.Dataset object from encoded list of words:
encoded_dataset_train = encoded_list_to_dataset(encoded_list_train, BATCH_SIZE, padded_shapes)

# VALIDATION: -------------------------------------------------

if not os.path.isfile(r'./encoded_list_validation.pickle'):
    # Get the encoded list of words for the validation set for the first time:
    encoded_list_val = gen_to_encoded_list(val_np, vocab, split='validation')
    print("\nvalidation set encoded!")
else:
    # Load encoded list of words from pickle:
    with open(r'./snli_pickles/encoded_list_val.pickle', 'rb') as pickle_file:
        encoded_list_val = pickle.load(pickle_file)

# Create tf.data.Dataset object from encoded list of words:
encoded_dataset_val = encoded_list_to_dataset(encoded_list_val, BATCH_SIZE, padded_shapes)

# TEST: ---------------------------------------------------------

if not os.path.isfile(r'./encoded_list_test.pickle'):
    # Get the encoded list of words for the test set for the first time:
    encoded_list_test = gen_to_encoded_list(test_np, vocab, split='test')
    print("test set encoded!\n")
else:
    # Load encoded list of words from pickle:
    with open(r'./encoded_list_test.pickle', 'rb') as pickle_file:
        encoded_list_test = pickle.load(pickle_file)

# Create tf.data.Dataset object from encoded list of words:
encoded_dataset_test = encoded_list_to_dataset(encoded_list_test, BATCH_SIZE, padded_shapes)


# Create the model -----------------------------------------------------------------------------------------------------


class ConvModel(tf.keras.Model):
    """ The model consist of two different convolutional paths for the two different type of senteces found in the snli
    dataset: hypotheses and premises.
    Then a concatenation of both (that we may want to try and extend in depth, for this may be the part where the
    inference relations are more strongly defined/ consolidated), and the final prediction layer for the 3 classes:
    entailment, neutral or contradiction.

    - 'name_hn' are names for Hypothesis sentences layers and ops
    - name_pn' are names for Premises sentences layers and ops.
    """

    def __init__(self, vocab, embedding_size, input_length=None):
        super(ConvModel, self).__init__()

        self.vocab_size = len(vocab) + 1
        self.embedding_size = embedding_size
        self.input_length = input_length

        # Shared Embedding layer for the vocabulary will be shared:
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, mask_zero=True,
                                                   input_length=input_length)
        # input_length: Length of input sequences, when it is constant. This argument is required if you are going to
        # connect Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed).

        # Separate paths for hypothesis sentences and premises sentences:
        # Hypotheses:
        self.conv_h1 = tf.keras.layers.Conv1D(32, 5, padding='same')
        self.conv_h2 = tf.keras.layers.Conv1D(64, 3, padding='same')
        self.conv_h3 = tf.keras.layers.Conv1D(96, 3, padding='same')

        self.activation_h1 = tf.keras.layers.Activation("relu")
        self.activation_h2 = tf.keras.layers.Activation("relu")
        self.activation_h3 = tf.keras.layers.Activation("relu")
        self.activation_h4 = tf.keras.layers.Activation("relu")

        # Premises:
        self.conv_p1 = tf.keras.layers.Conv1D(32, 5, padding='same')
        self.conv_p2 = tf.keras.layers.Conv1D(64, 3, padding='same')
        self.conv_p3 = tf.keras.layers.Conv1D(96, 3, padding='same')

        self.activation_p1 = tf.keras.layers.Activation("relu")
        self.activation_p2 = tf.keras.layers.Activation("relu")
        self.activation_p3 = tf.keras.layers.Activation("relu")
        self.activation_p4 = tf.keras.layers.Activation("relu")

        self.maxpooling_h1 = tf.keras.layers.MaxPooling1D()
        self.maxpooling_h2 = tf.keras.layers.MaxPooling1D()
        self.maxpooling_h3 = tf.keras.layers.MaxPooling1D()

        self.maxpooling_p1 = tf.keras.layers.MaxPooling1D()
        self.maxpooling_p2 = tf.keras.layers.MaxPooling1D()
        self.maxpooling_p3 = tf.keras.layers.MaxPooling1D()

        self.batch_norm_h1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_h2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_h3 = tf.keras.layers.BatchNormalization()

        self.batch_norm_p1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_p2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_p3 = tf.keras.layers.BatchNormalization()

        # padding for increasing num_filters dimensionality:
        self.paddings = tf.constant([[0, 0], [0, 0], [0, 32]])

        self.dropout_h1 = tf.keras.layers.Dropout(rate=0.8)
        self.dropout_h2 = tf.keras.layers.Dropout(rate=0.5)
        self.dropout_h3 = tf.keras.layers.Dropout(rate=0.3)

        self.dropout_p1 = tf.keras.layers.Dropout(rate=0.8)
        self.dropout_p2 = tf.keras.layers.Dropout(rate=0.5)
        self.dropout_p3 = tf.keras.layers.Dropout(rate=0.3)

        self.flatten_h = tf.keras.layers.Flatten()
        self.flatten_p = tf.keras.layers.Flatten()
        self.merged = tf.keras.layers.Concatenate()
        self.dense = tf.keras.layers.Dense(3, activation='softmax')

    def __call__(self, hypotheses_inputs, premises_inputs):
        
        # The embedding will share the weights for all the inputs (same embedding layer, different inputs):
        embedding_h = self.embedding(hypotheses_inputs)
        embedding_p = self.embedding(premises_inputs)

        # CONVOLUTIONAL PATH 1:
        x_1 = self.conv_h1(embedding_h)
        x_1 = self.batch_norm_h1(x_1)
        x_1 = self.activation_h1(x_1)
        x_1 = self.maxpooling_h1(x_1)
        x_1_output = self.dropout_h1(x_1)

        # padding to match the increasing num_filters dim between blocks:
        x_1_output_padded = tf.pad(x_1_output, self.paddings, 'CONSTANT')

        x_1_input = self.conv_h2(x_1_output)
        x_1 = self.batch_norm_h2(x_1_input)
        x_1 = self.activation_h2(x_1 + x_1_output_padded)  # Residual connection
        x_1 = self.maxpooling_h2(x_1)
        x_1_output = self.dropout_h2(x_1)

        # padding to match the increasing num_filters dim between blocks:
        x_1_output_padded = tf.pad(x_1_output, self.paddings, 'CONSTANT')

        x_1_input = self.conv_h3(x_1_output)
        x_1 = self.batch_norm_h3(x_1_input)
        x_1 = self.activation_h4(x_1 + x_1_output_padded)  # Residual connection
        x_1 = self.maxpooling_h3(x_1)
        x_1_final = self.dropout_h3(x_1)

        # CONVOLUTIONAL PATH 2:
        x_2_input = self.conv_p1(embedding_p)
        x_2 = self.batch_norm_p1(x_2_input)
        x_2 = self.activation_p1(x_2)
        x_2 = self.maxpooling_p1(x_2)
        x_2_output = self.dropout_p1(x_2)

        # padding to match the increasing num_filters dim between blocks:
        x_2_output_padded = tf.pad(x_2_output, self.paddings, 'CONSTANT')

        x_2_input = self.conv_p2(x_2_output)
        x_2 = self.batch_norm_p2(x_2_input)
        x_2 = self.activation_p2(x_2 + x_2_output_padded)  # Residual connection
        x_2 = self.maxpooling_p2(x_2)
        x_2_output = self.dropout_p2(x_2)

        # padding to match the increasing num_filters dim between blocks:
        x_2_output_padded = tf.pad(x_2_output, self.paddings, 'CONSTANT')

        x_2_input = self.conv_p3(x_2_output)
        x_2 = self.batch_norm_p3(x_2_input)
        x_2 = self.activation_p4(x_2 + x_2_output_padded)  # Residual connection
        x_2 = self.maxpooling_p3(x_2)
        x_2_final = self.dropout_p3(x_2)

        # Flatten and Concatenate both inputs:
        x_1_flatten = self.flatten_h(x_1_final)
        x_2_flatten = self.flatten_p(x_2_final)
        merged = self.merged([x_1_flatten, x_2_flatten])

        # Both outputs:
        output = self.dense(merged)

        return output


# Choose an optimizer and loss function for training:
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Possible improvement: trying a scheduled lr

# Select metrics to measure and print the loss and the accuracy of the model during the training loop
# (not internally used for optimization):

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Training step and loop -----------------------------------------------------------------------------------------------


def train_step(model, hypotheses_inputs, premises_inputs, labels):

    with tf.GradientTape() as tape:
        predictions = model(hypotheses_inputs, premises_inputs)
        current_loss = loss_object(labels, predictions)

    # Compute the gradient/derivative of arg_1 with respect to arg_2:
    gradients = tape.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Compute the overall (mean of losses) loss for this training batch/step:
    train_loss(current_loss)
    # Compute accuracy:
    train_accuracy(labels, predictions)


# Initialize the model:
conv_model = ConvModel(vocab, EMBEDDING_DIM, input_length=MAX_LEN)

# Checkpoints ----------------------------------------------------------------------------------------------------------
checkpoint_path = r"./checkpoints/train"                                  # "./checkpoints/convolutional_nlp_inference/3xconv_residual/train"
ckpt = tf.train.Checkpoint(model=conv_model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

# ----------------------------------------------------------------------------------------------------------------------

# Create summary writer for tensorboard:
current_time = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
train_log_dir = './logs/nlp_inference/3xconv_residual/train_' + current_time
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# ----------------------------------------------------------------------------------------------------------------------

print("\nHere we go into the training loop!")

EPOCHS = 5

for epoch in range(EPOCHS):

    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for batch, dataset_batch in enumerate(encoded_dataset_train):

        hypotheses_inputs, premises_inputs, labels = dataset_batch['hypotheses'], dataset_batch['premises'], \
                                                     dataset_batch['labels']

        train_step(conv_model, hypotheses_inputs, premises_inputs, labels)

        # if batch % 50 == 0:
        print('\nEpoch {} Batch {} => Nº Elements {} | Loss {:.4f} Accuracy {:.4f}'
                  .format(epoch + 1, batch + 1, (batch + 1) * 500, train_loss.result(), train_accuracy.result()))

        # Writes summaries for tensorboard:
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=batch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=batch)

    # Save checkpoints each epoch:
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    print('-' * 80)
    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
    print('Time taken for 1 epoch: {} secs/n; {} min'.format(time.time() - start, (time.time() - start)/60))


# Load last checkpoint for the model: ----------------------------------------------------------------------------------
last_checkpoint_path = "./checkpoints/train/ckpt-1"
checkpoint_dir = os.path.dirname(last_checkpoint_path)

latest = tf.train.latest_checkpoint(checkpoint_dir)
ckpt.restore(latest)
print(f'Latest checkpoint at: {latest} restored!!')

print()
print("-" * 80)
print("VALIDATION SET PREDICTIONS:")


# Evaluate on validation set:
for batch, dataset_batch in enumerate(encoded_dataset_val):

    hypotheses_inputs, premises_inputs, labels = dataset_batch['hypotheses'], dataset_batch['premises'], \
                                                     dataset_batch['labels']

    predictions = conv_model(hypotheses_inputs, premises_inputs)
    current_loss = loss_object(labels, predictions)

    val_loss(current_loss)
    val_accuracy(labels, predictions)

    print('\nBatch {} => Nº Elements {} | Loss {:.4f} Accuracy {:.4f}'
              .format(batch + 1, (batch + 1) * 500, val_loss.result(), val_accuracy.result()))

print()
print("-" * 80)
print("TEST SET PREDICTIONS:")

# Evaluate on test set:
for batch, dataset_batch in enumerate(encoded_dataset_test):

    hypotheses_inputs, premises_inputs, labels = dataset_batch['hypotheses'], dataset_batch['premises'], \
                                                 dataset_batch['labels']

    predictions = conv_model(hypotheses_inputs, premises_inputs)
    current_loss = loss_object(labels, predictions)

    test_loss(current_loss)
    test_accuracy(labels, predictions)

    print('\nBatch {} => Nº Elements {} | Loss {:.4f} Accuracy {:.4f}'
              .format(batch + 1, (batch + 1) * 500, test_loss.result(), test_accuracy.result()))
