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
			sell.cell_gen = rnn_cell.BasicRNNCell(args.rnn_size)
			sell.cell_dis = rnn_cell.BasicRNNCell(args.rnn_size)
		if args.model == 'gru':
			sell.cell_gen = rnn_cell.GRUCell(args.rnn_size)
			sell.cell_dis = rnn_cell.GRUCell(args.rnn_size)
		if args.model == 'lstm':
			sell.cell_gen = rnn_cell.BasicLSTMCell(args.rnn_size)
			sell.cell_dis = rnn_cell.BasicLSTMCell(args.rnn_size)
		else:
			raise Exception('model type not supported: {}'.format(args.model))

		sell.cell_gen = rnn_cell.MultiRNNCell([sell.cell_gen] * args.num_layers)
		sell.cell_dis = rnn_cell.MultiRNNCell([sell.cell_dis] * args.num_layers)

		# TODO 
		# Generate self.input_data to the Discriminator
		# 1.  Figure out how to do a batch of generated reviews in TF
		#     a. Tie in sample_distribution the generator
		# 2.  Use this batch of generated reviews to get the probabilities
		# 3.  Pass this as self.input_data, ensuring differentiability


		# Pass a tensor of probabilities over the characters to the model
		self.input_data    = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.vocab_size])
		self.targets       = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
		self.initial_state = sell.cell_dis.zero_state(args.batch_size, tf.float32)

		with tf.variable_scope('rnn'):
			softmax_w = tf.get_variable('softmax_w', [args.rnn_size, 2], trainable = False)
			softmax_b = tf.get_variable('softmax_b', [2], trainable = False)

			embedding = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])
			inputs    = tf.split(1, args.seq_length, self.input_data)
			inputs    = [tf.matmul(tf.squeeze(i, [1]), embedding) for i in inputs]

			self.inputs = inputs

			state   = self.initial_state
			outputs = []

			for i, inp in enumerate(inputs):
				if i > 0:
					tf.get_variable_scope().reuse_variables()
				output, state = sell.cell_dis(inp, state)
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

