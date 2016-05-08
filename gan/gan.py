import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import seq2seq
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

class GAN(object):
	def __init__(self, args, is_training=True):
		self.args = args

		if not is_training:
			args.batch_size = 1
			args.seq_length = 1

		# TODO
		# Set all variables within cell_dis to be trainable=False
		if args.model == 'rnn':
			self.cell_gen = rnn_cell.BasicRNNCell(args.rnn_size)
			self.cell_dis = rnn_cell.BasicRNNCell(args.rnn_size)
		if args.model == 'gru':
			self.cell_gen = rnn_cell.GRUCell(args.rnn_size)
			self.cell_dis = rnn_cell.GRUCell(args.rnn_size)
		if args.model == 'lstm':
			self.cell_gen = rnn_cell.BasicLSTMCell(args.rnn_size)
			self.cell_dis = rnn_cell.BasicLSTMCell(args.rnn_size)
		else:
			raise Exception('model type not supported: {}'.format(args.model))

		self.cell_gen = rnn_cell.MultiRNNCell([self.cell_gen] * args.num_layers)
		self.cell_dis = rnn_cell.MultiRNNCell([self.cell_dis] * args.num_layers)

		# TODO 
		# Generate self.input_data to the Discriminator
		# 2.  Use this batch of generated reviews to get the probabilities
		# 3.  Pass this as self.input_data, ensuring differentiability

		# Pass the generated sequences and targets (1)
		self.input_data  = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
		self.targets     = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

		# Both generator and discriminator should start with 0-states
		self.initial_state_gen = self.cell_gen.zero_state(args.batch_size, tf.float32)
		self.initial_state_dis = self.cell_dis.zero_state(args.batch_size, tf.float32)

		############
		# Generator
		############
		with tf.variable_scope('rnn_generator'):
			softmax_w = tf.get_variable('softmax_w', [args.rnn_size, args.vocab_size])
			softmax_b = tf.get_variable('softmax_b', [args.vocab_size])
			
			with tf.device('/cpu:0'):
				embedding = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])
				inputs_gen    = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
				inputs_gen    = [tf.squeeze(i, [1]) for i in inputs_gen]

		def loop(prev, _):
			prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
			prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
			return tf.nn.embedding_lookup(embedding, prev_symbol)

		outputs, last_state = seq2seq.rnn_decoder(inputs_gen, self.initial_state_gen, 
			self.cell_gen, loop_function=None if is_training else loop, scope='rnn_generator')
		output      = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
		self.logits_gen = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
		
		# TODO:
		#  Check appropriate dimensions:  
		#  [args.batch_size, args.seq_length, args.vocab_size]
		self.gen_probs  = tf.nn.softmax(self.logits_gen)
		# TODO
		# Check this reshape is being used correctly (probably is not)
		self.gen_probs  = tf.reshape(self.gen_probs, [args.batch_size, args.seq_length, args.vocab_size])

		################
		# Discriminator
		################
		# Pass a tensor of *probabilities* over the characters to the Discriminator
		with tf.variable_scope('rnn_discriminator'):
			softmax_w = tf.get_variable('softmax_w', [args.rnn_size, 2], trainable = False)
			softmax_b = tf.get_variable('softmax_b', [2], trainable = False)

			with tf.device('/cpu:0'):
				embedding = tf.get_variable('embedding', [args.vocab_size, args.rnn_size], trainable = False)
				
				# TODO:
				# Create appropriate inputs, the probability sequences from Generator
				inputs_dis    = tf.split(1, args.seq_length, self.gen_probs)
				inputs_dis    = [tf.matmul(tf.squeeze(i, [1]), embedding) for i in inputs_dis]

			self.inputs_dis = inputs_dis

			state   = self.initial_state_dis
			outputs = []

			for i, inp in enumerate(inputs_dis):
				if i > 0:
					tf.get_variable_scope().reuse_variables()
				output, state = self.cell_dis(inp, state)
				outputs.append(output)
			last_state = state

			output_tf   = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
			self.logits = tf.nn.xw_plus_b(output_tf, softmax_w, softmax_b)
			self.probs  = tf.nn.softmax(self.logits)

			loss = seq2seq.sequence_loss_by_example(
				[self.logits],
				[tf.reshape(self.targets, [-1])], 
				[tf.ones([args.batch_size * args.seq_length])],
				2)

			self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

			self.final_state = last_state
			self.lr          = tf.Variable(0.0, trainable = False)
			tvars 	         = tf.trainable_variables()
			grads, _         = tf.clip_by_global_norm(tf.gradients(self.cost, tvars, aggregation_method = 2), args.grad_clip)
			optimizer        = tf.train.AdamOptimizer(self.lr)
			self.train_op    = optimizer.apply_gradients(zip(grads, tvars))


	def train_discriminator(self):
		'''Train the discriminator classically'''
		pass

	def train_generator(self):
		'''Train the generator via adversarial training'''
		pass
		
	def generate_samples(self, sess, args, chars, vocab, seq_length = 200, initial = ' ', datafile = 'data/generated/test.txt'):
		''' Generate a batch of reviews entirely within TensorFlow'''		
		state = self.cell_gen.zero_state(args.batch_size, tf.float32).eval()

		sequence_matrix = []
		for i in xrange(args.batch_size):
			sequence_matrix.append([])
		char_arr = args.batch_size * [initial]
		
		probs_tf  = tf.placeholder(tf.float32, [args.batch_size, args.vocab_size])
		sample_op = self.sample_probs(probs_tf)

		for n in xrange(seq_length):
			x = np.zeros((args.batch_size, 1))
			for i, char in enumerate(char_arr):
				x[i,0] = vocab[char]    
			feed = {self.input_data: x, self.initial_state: state} 
			[probs, state] = sess.run([self.probs, self.final_state], feed)
			
			# Numpy implementation:
			sample_indexes = [int(np.random.choice(len(p), p=p)) for p in probs]
			print len(sample_indexes)
			char_arr = [chars[i] for i in sample_indexes]
			for i, char in enumerate(char_arr):
				sequence_matrix[i].append(char)
		
		with open(datafile, 'wb') as f:
			for line in sequence_matrix:
				print>>f, ''.join(line) 



