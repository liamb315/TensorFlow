import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import seq2seq


class Generator(object):
    def __init__(self, args, is_training=True):
        self.args = args

        if not is_training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell = rnn_cell.BasicRNNCell(args.rnn_size)
        elif args.model == 'gru':
            cell = rnn_cell.GRUCell(args.rnn_size)
        elif args.model == 'lstm':
            cell = rnn_cell.BasicLSTMCell(args.rnn_size)
        else:
            raise Exception("model type not supported: {}".format(args.model))

        self.cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data     = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets        = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state  = self.cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnn'):
            softmax_w = tf.get_variable('softmax_w', [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable('softmax_b', [args.vocab_size])
            
            with tf.device('/cpu:0'):
                embedding = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])
                inputs    = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs    = [tf.squeeze(i, [1]) for i in inputs]

        def loop(prev, _):
            prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, 
            self.cell, loop_function=None if is_training else loop, scope='rnn')
        output      = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        self.probs  = tf.nn.softmax(self.logits)
        loss        = seq2seq.sequence_loss_by_example([self.logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([args.batch_size * args.seq_length])],
            args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr       = tf.Variable(0.0, trainable = False)
        tvars         = tf.trainable_variables()
        grads, _      = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        optimizer     = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, chars, vocab, seq_length = 200, initial=''):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in initial[:-1]:
            x       = np.zeros((1,1))
            x[0,0]  = vocab[char]
            feed    = {self.input_data: x, self.intitial_state: state}
            [state] = sess.run([self.final_state], feed)

        sequence = initial
        char = initial[-1]
        for n in xrange(seq_length):
            x = np.zeros((1,1))
            x[0,0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p  = probs[0]
            sample = int(np.random.choice(len(p), p=p))
            pred = chars[sample]
            sequence += pred
            char = pred
        return sequence


class Discriminator(object):
    def __init__(self, args, is_training=True):
        self.args = args

        if not is_training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            self.cell = rnn_cell.BasicRNNCell(args.rnn_size)
        if args.model == 'gru':
            self.cell = rnn_cell.GRUCell(args.rnn_size)
        if args.model == 'lstm':
            self.cell = rnn_cell.BasicLSTMCell(args.rnn_size)
        else:
            raise Exception('model type not supported: {}'.format(args.model))

        self.cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input         = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets       = tf.placeholder(tf.int32, [args.batch_size, 1])
        self.initial_state = tf.placeholder(args.batch_size, tf.float32)

        with tf.variable_scope('rnn_d'):
            softmax_w = tf.get_variable('softmax_w', [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable('softmax_b' [args.vocab_size])
