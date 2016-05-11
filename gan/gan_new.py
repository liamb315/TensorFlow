import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import seq2seq
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from argparse import ArgumentParser


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='data',
		help='data directory containing reviews')
	parser.add_argument('--save_dir_GAN', type=str, default='models_GAN',
		help='directory to store checkpointed GAN models')
	parser.add_argument('--rnn_size', type=int, default=128,
		help='size of RNN hidden state')
	parser.add_argument('--num_layers', type=int, default=2,
		help='number of layers in the RNN')
	parser.add_argument('--model', type=str, default='lstm',
		help='rnn, gru, or lstm')
	parser.add_argument('--batch_size', type=int, default=5,
		help='minibatch size')
	parser.add_argument('--seq_length', type=int, default=20,
		help='RNN sequence length')
	parser.add_argument('-n', type=int, default=500,
		help='number of characters to sample')
	parser.add_argument('--prime', type=str, default=' ',
		help='prime text')
	parser.add_argument('--num_epochs_GAN', type=int, default=5,
		help='number of epochs to train GAN')
	parser.add_argument('--save_every', type=int, default=50,
		help='save frequency')
	parser.add_argument('--grad_clip', type=float, default=5.,
		help='clip gradients at this value')
	parser.add_argument('--learning_rate_gen', type=float, default=0.002,
		help='learning rate')
	parser.add_argument('--learning_rate_dis', type=float, default=0.0002,
		help='learning rate for discriminator')
	parser.add_argument('--decay_rate', type=float, default=0.97,
		help='decay rate for rmsprop')
	parser.add_argument('--keep_prob', type=float, default=0.5,
		help='keep probability for dropout')
	parser.add_argument('--vocab_size', type=float, default=100,
		help='size of the vocabulary (characters)')
	return parser.parse_args()


def minimize_and_clip(optimizer, objective, var_list, clip_val=5):
	gradients = optimizer.compute_gradients(objective, var_list=var_list)
	for i, (grad, var) in enumerate(gradients):
		if grad is not None:
			gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
	return optimizer.apply_gradients(gradients)


def generator(input_data, args, reuse=False):
	'''
	Produce a probability sequence from the provided input_sequence

	args:
		input_data:   
		args:  

	returns:
		probs:   [args.batch_size, args.seq_length, args.vocab_size]

	'''
	with tf.variable_scope('generator', args, reuse = reuse):
		if args.model == 'rnn':
			cell = rnn_cell.BasicRNNCell(args.rnn_size)
		if args.model == 'gru':
			cell = rnn_cell.GRUCell(args.rnn_size)
		if args.model == 'lstm':
			cell = rnn_cell.BasicLSTMCell(args.rnn_size)
		else:
			raise Exception('model type not supported: {}'.format(args.model))
		cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)
		initial_state = cell.zero_state(args.batch_size, tf.float32)

		with tf.variable_scope('rnn'):
			softmax_w = tf.get_variable('softmax_w', [args.rnn_size, args.vocab_size])
			softmax_b = tf.get_variable('softmax_b', [args.vocab_size])
			
			with tf.device('/cpu:0'):
				embedding  = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])
				inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, input_data))
				inputs = [tf.squeeze(i, [1]) for i in inputs]

		def loop(prev, _):
			prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
			prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
			return tf.nn.embedding_lookup(embedding, prev_symbol)

		outputs, last_state = seq2seq.rnn_decoder(inputs, initial_state, cell, 
				loop_function=None if is_training else loop, scope='rnn')
		
		#  Dim: [args.batch_size * args.seq_length, args.rnn_size]
		output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
		#  Dim: [args.batch_size * args.seq_length, args.vocab_size]
		logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
		probs  = tf.nn.softmax(logits)
		# Dim:  [args.batch_size, args.seq_length, args.vocab_size]
		probs  = tf.reshape(probs, [args.batch_size, args.seq_length, args.vocab_size])
		return probs


def discriminator(input_sequence, args, reuse=False):
	with tf.variable_scope('discriminator', reuse = reuse):
		if args.model == 'rnn':
			cell = rnn_cell.BasicRNNCell(args.rnn_size)
		if args.model == 'gru':
			cell = rnn_cell.GRUCell(args.rnn_size)
		if args.model == 'lstm':
			cell = rnn_cell.BasicLSTMCell(args.rnn_size)
		else:
			raise Exception('model type not supported: {}'.format(args.model))
		cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)
		initial_state = cell.zero_state(args.batch_size, tf.float32)

		with tf.variable_scope('rnn'):
			softmax_w = tf.get_variable('softmax_w', [args.rnn_size, 2])
			softmax_b = tf.get_variable('softmax_b', [2])

			with tf.device('/cpu:0'):
				embedding  = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])
				inputs     = tf.split(1, args.seq_length, input_sequence)
				inputs     = [tf.matmul(tf.squeeze(inp, [1]), embedding) for inp in inputs]

			state   = initial_state
			outputs = []

			for i, inp in enumerate(inputs):
				if i > 0:
					tf.get_variable_scope().reuse_variables()
				output, state = cell(inp, state)
				outputs.append(output)
			last_state = state
			output_tf   = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
			logits = tf.nn.xw_plus_b(output_tf, softmax_w, softmax_b)
			probs  = tf.nn.softmax(logits)
			return probs, logits

def train_generator():
	pass

def train_discriminator():
	pass


if __name__ == '__main__':
	ops.reset_default_graph()       
	if 'session' in globals():         
		session.close()                
	session = tf.Session()             
	args = parse_args()

	# Generator Training
	input_data    = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
	targts        = tf.placeholder(tf.int32, [args.batch_size, args.seq_length]) # Should be 1 for real
	gen_seq       = generator(input_data, args)
	gen_loss      = seq2seq.sequence_loss_by_example(
					[discriminator(gen_seq, args)[1]], # Input wants logits, not probs
					[tf.reshape(targets, [-1])], 
					[tf.ones([args.batch_size * args.seq_length])],
					2)
	gen_cost      = tf.reduce_sum(gen_loss) / args.batch_size / args.seq_length
	gen_vars      = [v for v in tf.all_variables() if v.name.startswith("generator/")]
	gen_optimizer = tf.train.AdamOptimizer(args.learning_rate_gen)
	gen_train_op  = minimize_and_clip(gen_optimizer, objective = gen_cost, var_list = gen_vars)


	# Discriminator Training
	# TODO:  Should this be tf.int32?
	input_real_seq  = tf.placholder(tf.float32, [args.batch_size, args.seq_length, args.vocab_size]) 
	input_gen_seq   = tf.placholder(tf.float32, [args.batch_size, args.seq_length, args.vocab_size]) 

	dis_real_prob   = discriminator(input_real_seq, args)
	dis_fake_prob   = discriminator(input_gen_seq, args)

	# DISCRIMIN_BATCH = 128 
	# dis_real_image = tf.placeholder(tf.float32, (DISCRIMIN_BATCH, 64, 64, 3))
	# dis_z          = tf.placeholder(tf.float32, (DISCRIMIN_BATCH, GENERATOR_SEED,))
	# dis_real_prob  = discrimin(dis_real_image, reuse=True)
	# dis_gen_prob   = discrimin(generator(dis_z, reuse=True), reuse=True)
	# dis_score      = tf.log(dis_real_prob) + tf.log(1. - dis_gen_prob)
	# dis_score      = tf.reduce_mean(dis_score)
	dis_vars      = [v for v in tf.all_variables() if v.name.startswith("discriminator/")]
	dis_optimizer  = tf.train.AdamOptimizer(args.learning_rate_dis)
	# dis_op         = minimize_and_clip(dis_optimizer, objective=-dis_score, var_list=discrimin_vars)


	try:
		print generator_vars
		# for epoch in range(START_EPOCH, 10):
		#     for next_idx, batch in batched_images(START_IDX):
		#         START_EPOCH, START_IDX = epoch, next_idx
		#         batch_start_time = time.time()
		#         dis_err = train_discrimin(batch)
		#         gen_err = train_generator()
		#         dis_graph.append(dis_err)
		#         gen_graph.append(gen_err)
		#         batch_time = time.time() - batch_start_time
		#         clear_output(wait=True)
		#         print("Epoch %d: image %d (%.1f images/s)" % (epoch, next_idx, DISCRIMIN_BATCH / batch_time), flush=True)
		#     saver.save(session, "./saved_models/lsun-epoch%d.ckpt" % (epoch,))
		#     START_IDX = 0
	except KeyboardInterrupt:
		print("Interrupted")