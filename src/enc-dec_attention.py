import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
import time
import matplotlib.pyplot as plt
import pickle


from data_util import read_dataset


# read dataset
dataset_location = "/Users/sjayakum/Desktop/nlp-code-review/data/data.p"
X, Y, en_word2idx, en_idx2word, en_vocab, hi_word2idx, hi_idx2word, hi_vocab = read_dataset(dataset_location)



def data_padding(x, y, length = 20):
    for i in range(len(x)):
        x[i] = x[i] + (length - len(x[i])) * [en_word2idx['<pad>']]
        y[i] = [hi_word2idx['<go>']] + y[i] + [hi_word2idx['<eos>']] + (length-len(y[i])) * [hi_word2idx['<pad>']]

data_padding(X, Y)

# data splitting
X_train,  X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1)



# build a model

input_seq_len = 20
output_seq_len = 22
en_vocab_size = len(en_vocab) + 2 # + <pad>, <ukn>
hi_vocab_size = len(hi_vocab) + 4 # + <pad>, <ukn>, <eos>, <go>

# placeholders
encoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'encoder{0}'.format(i)) for i in range(input_seq_len)]
decoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'decoder{0}'.format(i)) for i in range(output_seq_len)]

targets = [decoder_inputs[i+1] for i in range(output_seq_len-1)]
# add one more target
targets.append(tf.placeholder(dtype = tf.int32, shape = [None], name = 'last_target'))
target_weights = [tf.placeholder(dtype = tf.float32, shape = [None], name = 'target_w{0}'.format(i)) for i in range(output_seq_len)]

# output projection
size = 512
w_t = tf.get_variable('proj_w', [hi_vocab_size, size], tf.float32)
b = tf.get_variable('proj_b', [hi_vocab_size], tf.float32)
w = tf.transpose(w_t)
output_projection = (w, b)

outputs, states = tf.nn.seq2seq.embedding_attention_seq2seq(
                                            encoder_inputs,
                                            decoder_inputs,
                                            tf.nn.rnn_cell.BasicLSTMCell(size),
                                            num_encoder_symbols = en_vocab_size,
                                            num_decoder_symbols = hi_vocab_size,
                                            embedding_size = 80,
                                            feed_previous = False,
                                            output_projection = output_projection,
                                            dtype = tf.float32)


def sampled_loss(logits, labels):
    return tf.nn.sampled_softmax_loss(
                        weights = w_t,
                        biases = b,
                        labels = tf.reshape(labels, [-1, 1]),
                        inputs = logits,
                        num_sampled = 512,
                        num_classes = hi_vocab_size)

# Weighted cross-entropy loss for a sequence of logits
loss = tf.nn.seq2seq.sequence_loss(outputs, targets, target_weights, softmax_loss_function = sampled_loss)


def softmax(x):
    n = np.max(x)
    e_x = np.exp(x - n)
    return e_x / e_x.sum()


# feed data into placeholders
def feed_dict(x, y, batch_size=64):
    feed = {}

    idxes = np.random.choice(len(x), size=batch_size, replace=False)

    for i in range(input_seq_len):
        feed[encoder_inputs[i].name] = np.array([x[j][i] for j in idxes])

    for i in range(output_seq_len):
        feed[decoder_inputs[i].name] = np.array([y[j][i] for j in idxes])

    feed[targets[len(targets) - 1].name] = np.full(shape=[batch_size], fill_value=hi_word2idx['<pad>'])

    for i in range(output_seq_len - 1):
        batch_weights = np.ones(batch_size, dtype=np.float32)
        target = feed[decoder_inputs[i + 1].name]
        for j in range(batch_size):
            if target[j] == hi_word2idx['<pad>']:
                batch_weights[j] = 0.0
        feed[target_weights[i].name] = batch_weights

    feed[target_weights[output_seq_len - 1].name] = np.zeros(batch_size, dtype=np.float32)

    return feed


# decode output sequence
def decode_output(output_seq):
    words = []
    for i in range(output_seq_len):
        smax = softmax(output_seq[i])
        idx = np.argmax(smax)
        words.append(hi_idx2word[idx])
    return words


# ops and hyperparameters
learning_rate = 3e-3
batch_size = 8
steps = 30000

# ops for projecting outputs
outputs_proj = [tf.matmul(outputs[i], output_projection[0]) + output_projection[1] for i in range(output_seq_len)]

# training op
optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.99).minimize(loss)

# init op
init = tf.global_variables_initializer()

# forward step
def forward_step(sess, feed):
    output_sequences = sess.run(outputs_proj, feed_dict = feed)
    return output_sequences

# training step
def backward_step(sess, feed):
    sess.run(optimizer, feed_dict = feed)


# we will use this list to plot losses through steps
losses = []

# save a checkpoint so we can restore the model later
saver = tf.train.Saver()

print('------------------TRAINING------------------')

with tf.Session() as sess:
    sess.run(init)

    t = time.time()
    for step in range(steps):
        feed = feed_dict(X_train, Y_train)

        backward_step(sess, feed)

        if step % 50 == 49 or step == 0:
            loss_value = sess.run(loss, feed_dict=feed)
            print('step: {}, loss: {}'.format(step, loss_value))
            losses.append(loss_value)

        if step % 1000 == 999:
            saver.save(sess, 'D:/NLP Project/Hindi English/checkpoints/', global_step=step)
            print('Checkpoint is saved')

    print('Training time for {} steps: {}s'.format(steps, time.time() - t))


with plt.style.context('fivethirtyeight'):
    plt.plot(losses, linewidth = 1)
    plt.xlabel('Steps')
    plt.ylabel('Losses')
    plt.ylim((0, 12))

plt.show()

with tf.Graph().as_default():
    # placeholders
    encoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[None], name='encoder{}'.format(i)) for i in
                      range(input_seq_len)]
    decoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[None], name='decoder{}'.format(i)) for i in
                      range(output_seq_len)]

    # output projection
    size = 512
    w_t = tf.get_variable('proj_w', [hi_vocab_size, size], tf.float32)
    b = tf.get_variable('proj_b', [hi_vocab_size], tf.float32)
    w = tf.transpose(w_t)
    output_projection = (w, b)

    # embedding size changed from 100 to 150

    # change the model so that output at time t can be fed as input at time t+1
    outputs, states = tf.nn.seq2seq.embedding_attention_seq2seq(
        encoder_inputs,
        decoder_inputs,
        tf.nn.rnn_cell.BasicLSTMCell(size),
        num_encoder_symbols=en_vocab_size,
        num_decoder_symbols=hi_vocab_size,
        embedding_size=80,
        feed_previous=True,  # <-----this is changed----->
        output_projection=output_projection,
        dtype=tf.float32)

    # ops for projecting outputs
    outputs_proj = [tf.matmul(outputs[i], output_projection[0]) + output_projection[1] for i in range(output_seq_len)]

    # let's translate these sentences
    # let's translate these sentences
    #     le_sentences = ["What' s your name", 'My name is', 'What are you doing', 'I am reading a book',\
    #                     'How are you', 'I am good', 'Do you speak English', 'What time is it', 'Hi', 'Goodbye', 'Yes', 'No']

    #     #le_sentences = en_vocab[:8]
    #     #le_sentences = ["This", "time", "what", "why", "how"]
    #     le_sentences = ["I", "Absolute position", "Here she wrote a rectangle", "I should have liked to begin this story in the fashion of the fairy-tales ."]
    #     en_sentences = []
    #     for each_sentence in le_sentences:
    #         en_sentences.append(each_sentence.lower())

    #     #en_sentences = en_vocab[:20]
    #     en_sentences_encoded = [[en_word2idx.get(word, 0) for word in en_sentence.split()] for en_sentence in en_sentences]

    #     # padding to fit encoder input
    #     for i in range(len(en_sentences_encoded)):
    #         en_sentences_encoded[i] += (20 - len(en_sentences_encoded[i])) * [en_word2idx['<pad>']]

    for idx in range(1, 11):

        en_sentences_encoded = X_test[(idx - 1) * 1000:(idx) * 1000]
        hi_sentences_encoded = Y_test[(idx - 1) * 1000:(idx) * 1000]
        en_sentences = []
        hi_sentences = []

        import codecs

        fp = codecs.open("D:/NLP Project/Hindi English/test_results" + str(idx) + ".txt", encoding="utf-8", mode="w")

        for j in range(len(en_sentences_encoded)):
            temp = ""
            for i in range(len(en_sentences_encoded[j])):
                temp += en_idx2word[en_sentences_encoded[j][i]] + " "
            en_sentences.append(temp)

        for j in range(len(hi_sentences_encoded)):
            temp = ""
            for i in range(len(hi_sentences_encoded[j])):
                temp += hi_idx2word[hi_sentences_encoded[j][i]] + " "
            hi_sentences.append(temp)

        # restore all variables - use the last checkpoint saved
        saver = tf.train.Saver()
        path = tf.train.latest_checkpoint('D:/NLP Project/Hindi English/checkpoints/')

        with tf.Session() as sess:
            # restore
            saver.restore(sess, path)

            # feed data into placeholders
            feed = {}
            for i in range(input_seq_len):
                feed[encoder_inputs[i].name] = np.array(
                    [en_sentences_encoded[j][i] for j in range(len(en_sentences_encoded))])

            feed[decoder_inputs[0].name] = np.array([hi_word2idx['<go>']] * len(en_sentences_encoded))

            # translate
            output_sequences = sess.run(outputs_proj, feed_dict=feed)

            # decode seq.
            for i in range(len(en_sentences_encoded)):
                fp.write('\n')
                fp.write('{}.\n--------------------------------'.format(i + 1))
                fp.write('\n')
                # print('{}.\n--------------------------------'.format(i+1))
                ouput_seq = [output_sequences[j][i] for j in range(output_seq_len)]
                # decode output sequence
                words = decode_output(ouput_seq)

                # print(en_sentences[i])
                # print('\n')
                fp.write('Input\t\t - ')
                actual_sentences = en_sentences[i].split()
                for j in range(len(actual_sentences)):
                    if actual_sentences[j] not in ['<eos>', '<pad>', '<go>']:
                        fp.write(actual_sentences[j] + " ")
                fp.write('\n')

                fp.write('Actual\t\t - ')
                actual_output = hi_sentences[i].split()
                for j in range(len(actual_output)):
                    if actual_output[j] not in ['<eos>', '<pad>', '<go>']:
                        fp.write(actual_output[j] + " ")
                fp.write('\n')

                fp.write('Predicted\t\t - ')
                for i in range(len(words)):
                    if words[i] not in ['<eos>', '<pad>', '<go>']:
                        # print(words[i], end=' ')
                        fp.write(words[i] + " ")

                # print('\n--------------------------------')
                fp.write('\n--------------------------------')
                fp.write('\n')
        fp.close()
print("DONE")