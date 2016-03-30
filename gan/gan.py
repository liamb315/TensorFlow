import tensorflow as tf
import numpy as np
import logging
from tensorflow.models.rnn import *
from argparse import ArgumentParser
from batcher import Batcher
from model import Generator
import time
import os
import cPickle

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='data',
		help='data directory containing reviews')
	parser.add_argument('--save_dir', type=str, default='models',
		help='directory to store checkpointed models')
	parser.add_argument('--rnn_size', type=int, default=256,
		help='size of RNN hidden state')
	parser.add_argument('--num_layers', type=int, default=2,
		help='number of layers in the RNN')
	parser.add_argument('--model', type=str, default='lstm',
		help='rnn, gru, or lstm')
	parser.add_argument('--batch_size', type=int, default=100,
		help='minibatch size')
	parser.add_argument('--seq_length', type=int, default=200,
		help='RNN sequence length')
	parser.add_argument('--num_epochs', type=int, default=50,
		help='number of epochs')
	parser.add_argument('--save_every', type=int, default=50,
		help='save frequency')
	parser.add_argument('--grad_clip', type=float, default=5.,
		help='clip gradients at this value')
	parser.add_argument('--learning_rate', type=float, default=0.002,
		help='learning rate')
	parser.add_argument('--decay_rate', type=float, default=0.97,
		help='decay rate for rmsprop')
	parser.add_argument('--keep_prob', type=float, default=0.5,
		help='keep probability for dropout')
	parser.add_argument('--vocab_size', type=float, default=100,
		help='size of the vocabulary (characters)')
	return parser.parse_args()


def train_generator(args):
	'''Train the generator via classical approach'''
	logging.debug('Batcher...')
	batcher   = Batcher(args.data_dir, args.batch_size, args.seq_length)

	logging.debug('Vocabulary...')
	with open(os.path.join(args.save_dir, 'config.pkl'), 'w') as f:
		cPickle.dump(args, f)
	with open(os.path.join(args.save_dir, 'real_beer_vocab.pkl'), 'w') as f:
		cPickle.dump((batcher.chars, batcher.vocab), f)

	logging.debug('Creating generator...')
	generator = Generator(args)

	with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
		tf.initialize_all_variables().run()
		saver = tf.train.Saver(tf.all_variables())

		for epoch in xrange(args.num_epochs):
			# Anneal learning rate
			new_lr_rate = args.learning_rate * (args.decay_rate ** epoch)
			sess.run(tf.assign(generator.lr, new_lr_rate))
			batcher.reset_batch_pointer()
			state = generator.initial_state.eval()

			for batch in xrange(batcher.num_batches):
				start = time.time()
				x, y  = batcher.next_batch()
				feed  = {generator.input_data: x, generator.targets: y, generator.initial_state: state}
				train_loss, state, _ = sess.run([generator.cost, generator.final_state, generator.train_op], feed)
				end   = time.time()
				print '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}' \
					.format(epoch * batcher.num_batches + batch,
						args.num_epochs * batcher.num_batches,
						epoch, train_loss, end - start)
				if (epoch * batcher.num_batches + batch) % args.save_every == 0:
					checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
					saver.save(sess, checkpoint_path, global_step = epoch * batcher.num_batches + batch)
					print 'Model saved to {}'.format(checkpoint_path)


def train_generator_interactive(args, sess, generator, batcher, num_epochs, num_batches, learning_rate=100):
	'''Train generator in an interactive environment'''
	# tf.initialize_all_variables().run()
	# saver = tf.train.Saver(tf.all_variables())

	for epoch in xrange(num_epochs):
		# Anneal learning rate
		new_lr_rate = learning_rate * (args.decay_rate ** epoch)
		sess.run(tf.assign(generator.lr, new_lr_rate))
		# batcher.reset_batch_pointer()
		state = generator.initial_state.eval()

		for batch in xrange(num_batches):
			start = time.time()
			x, y  = batcher.next_batch()
			feed  = {generator.input_data: x, generator.targets: y, generator.initial_state: state}
			train_loss, state, _ = sess.run([generator.cost, generator.final_state, generator.train_op], feed)
			end   = time.time()
			print '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}' \
				.format(epoch * batcher.num_batches + batch,
					num_epochs * batcher.num_batches,
					epoch, train_loss, end - start)
			if (epoch * batcher.num_batches + batch) % args.save_every == 0:
				checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step = epoch * batcher.num_batches + batch)
				print 'Model saved to {}'.format(checkpoint_path)


def sample_interactive(args, sess, generator, num_char = 500, prime = ' '):
	with open(os.path.join(args.save_dir, 'config.pkl')) as f:
		saved_args = cPickle.load(f)
	with open(os.path.join(args.save_dir, 'real_beer_vocab.pkl')) as f:
		chars, vocab = cPickle.load(f)
	model = Generator(saved_args, False)
	saver = tf.train.Saver(tf.all_variables())
	ckpt = tf.train.get_checkpoint_state(args.save_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print model.sample(sess, chars, vocab, num_char, prime)


if __name__=='__main__':	
	args = parse_args()
	# Standard TensorFlow Session
	# train_generator(args)

	#################################
	# Interactive TensorFlow Session
	#################################
	# sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
	# # tf.initialize_all_variables().run()

	# logging.debug('Batcher...')
	# batcher = Batcher(args.data_dir, args.batch_size, args.seq_length)
	
	# logging.debug('Vocabulary...')
	# with open(os.path.join(args.save_dir, 'config.pkl'), 'w') as f:
	# 	cPickle.dump(args, f)
	# with open(os.path.join(args.save_dir, 'real_beer_vocab.pkl'), 'w') as f:
	# 	cPickle.dump((batcher.chars, batcher.vocab), f)

	# logging.debug('Generator...')
	# generator = Generator(args)


	# train_generator_interactive(args, sess, generator, batcher, 1, 100)


	# #sess.close()