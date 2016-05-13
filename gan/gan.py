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

		if not is_training:
			seq_length = 1
		else:
			seq_length = args.seq_length

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

		with tf.variable_scope('generator'):
			self.cell_gen = rnn_cell.MultiRNNCell([self.cell_gen] * args.num_layers)
		with tf.variable_scope('discriminator'):
			self.cell_dis = rnn_cell.MultiRNNCell([self.cell_dis] * args.num_layers)

		# Pass the generated sequences and targets (1)
		self.input_data  = tf.placeholder(tf.int32, [args.batch_size, seq_length])
		self.targets     = tf.placeholder(tf.int32, [args.batch_size, seq_length])

		############
		# Generator
		############
		with tf.variable_scope('generator'):
			self.initial_state_gen = self.cell_gen.zero_state(args.batch_size, tf.float32)	

			with tf.variable_scope('rnn'):
				softmax_w = tf.get_variable('softmax_w', [args.rnn_size, args.vocab_size])
				softmax_b = tf.get_variable('softmax_b', [args.vocab_size])
				
				with tf.device('/cpu:0'):
					embedding  = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])
					inputs_gen = tf.split(1, seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
					inputs_gen = [tf.squeeze(i, [1]) for i in inputs_gen]

			def loop(prev, _):
				prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
				prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
				return tf.nn.embedding_lookup(embedding, prev_symbol)

			outputs_gen, last_state_gen = seq2seq.rnn_decoder(inputs_gen, self.initial_state_gen, 
				self.cell_gen, loop_function=None if is_training else loop, scope='rnn')
			
			#  Dim: [args.batch_size * seq_length, args.rnn_size]
			output_gen      = tf.reshape(tf.concat(1, outputs_gen), [-1, args.rnn_size])

			#  Dim: [args.batch_size * seq_length, args.vocab_size]
			self.logits_gen         = tf.nn.xw_plus_b(output_gen, softmax_w, softmax_b)
			self.probs_gen_grouped  = tf.nn.softmax(self.logits_gen)

			# Dim:  [args.batch_size, seq_length, args.vocab_size]
			self.probs_gen       = tf.reshape(self.probs_gen_grouped, [args.batch_size, seq_length, args.vocab_size])
			self.final_state_gen = last_state_gen

		################
		# Discriminator
		################
		# Pass a tensor of *probabilities* over the characters to the Discriminator
		with tf.variable_scope('discriminator'):
			self.initial_state_dis = self.cell_dis.zero_state(args.batch_size, tf.float32)

			with tf.variable_scope('rnn'):
				softmax_w = tf.get_variable('softmax_w', [args.rnn_size, 2])
				softmax_b = tf.get_variable('softmax_b', [2])

				with tf.device('/cpu:0'):
					embedding  = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])
					inputs_dis = tf.split(1, seq_length, self.probs_gen)
					inputs_dis = [tf.matmul(tf.squeeze(i, [1]), embedding) for i in inputs_dis]

				self.inputs_dis = inputs_dis
				state_dis   = self.initial_state_dis
				outputs_dis = []

				for i, inp in enumerate(inputs_dis):
					if i > 0:
						tf.get_variable_scope().reuse_variables()
					output_dis, state_dis = self.cell_dis(inp, state_dis)
					outputs_dis.append(output_dis)
				last_state_dis = state_dis

				output_tf   = tf.reshape(tf.concat(1, outputs_dis), [-1, args.rnn_size])
				self.logits = tf.nn.xw_plus_b(output_tf, softmax_w, softmax_b)
				self.probs  = tf.nn.softmax(self.logits)

		gen_loss = seq2seq.sequence_loss_by_example(
			[self.logits],
			[tf.reshape(self.targets, [-1])], 
			[tf.ones([args.batch_size * seq_length])],
			2)

		self.gen_cost = tf.reduce_sum(gen_loss) / args.batch_size / seq_length

		self.final_state_dis = last_state_dis
		self.lr_gen = tf.Variable(0.0, trainable = False)		
		
		if is_training:
			self.tvars 	= tf.trainable_variables()
			gen_vars             = [v for v in self.tvars if v.name.startswith("generator/")]
			gen_grads, _         = tf.clip_by_global_norm(tf.gradients(self.gen_cost, gen_vars, aggregation_method = 2), args.grad_clip)
			gen_optimizer        = tf.train.AdamOptimizer(self.lr_gen)
			self.gen_train_op    = gen_optimizer.apply_gradients(zip(gen_grads, gen_vars))				

		
	def generate_samples(self, sess, args, chars, vocab, seq_length = 200, initial = ' ', datafile = 'data/gan/fake_reviews.txt'):
		''' Generate a batch of reviews'''		
		state = self.cell_gen.zero_state(args.batch_size, tf.float32).eval()

		sequence_matrix = []
		for i in xrange(args.batch_size):
			sequence_matrix.append([])
		char_arr = args.batch_size * [initial]
		
		for n in xrange(seq_length):
			x = np.zeros((args.batch_size, 1))
			for i, char in enumerate(char_arr):
				x[i,0] = vocab[char]    
			feed = {self.input_data: x, self.initial_state_gen: state} 
			[probs, state] = sess.run([self.probs_gen_grouped, self.final_state_gen], feed)
			sample_indexes = [int(np.random.choice(len(p), p=p)) for p in probs]
			char_arr = [chars[i] for i in sample_indexes]
			for i, char in enumerate(char_arr):
				sequence_matrix[i].append(char)
		
		with open(datafile, 'wb') as f:
			for line in sequence_matrix:
				print ''.join(line)
				print>>f, ''.join(line) 

		return sequence_matrix