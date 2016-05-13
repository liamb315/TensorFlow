import tensorflow as tf
import numpy as np
import logging
from tensorflow.models.rnn import *
from argparse import ArgumentParser
from batcher import GANBatcher
from gan import GAN
from discriminator import Discriminator
import time
import os
import cPickle

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='data/gan',
		help='data directory containing reviews')
	parser.add_argument('--save_dir_GAN', type=str, default='models_GAN',
		help='directory to store checkpointed GAN models')
	parser.add_argument('--rnn_size', type=int, default=128,
		help='size of RNN hidden state')
	parser.add_argument('--num_layers', type=int, default=2,
		help='number of layers in the RNN')
	parser.add_argument('--model', type=str, default='lstm',
		help='rnn, gru, or lstm')
	parser.add_argument('--batch_size', type=int, default=50,
		help='minibatch size')
	parser.add_argument('--seq_length', type=int, default=200,
		help='RNN sequence length')
	parser.add_argument('-n', type=int, default=500,
		help='number of characters to sample')
	parser.add_argument('--prime', type=str, default=' ',
		help='prime text')
	parser.add_argument('--num_epochs', type=int, default=5,
		help='number of epochs')
	parser.add_argument('--num_epochs_GAN', type=int, default=1,
		help='number of epochs of GAN')
	parser.add_argument('--num_epochs_gen', type=int, default=5,
		help='number of epochs to train generator')
	parser.add_argument('--num_epochs_dis', type=int, default=5,
		help='number of epochs to train discriminator')
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
	parser.add_argument('--vocab_size', type=float, default=5,
		help='size of the vocabulary (characters)')
	return parser.parse_args()



def train_gan(args):
	'''Adversarial Training'''
	logging.debug('GAN batcher...')
	# TODO:  Write a proper batcher for GAN
	batcher  = GANBatcher(args.data_dir, args.batch_size, args.seq_length)

	logging.debug('Vocabulary...')
	with open(os.path.join(args.save_dir_GAN, 'config.pkl'), 'w') as f:
		cPickle.dump(args, f)
	with open(os.path.join(args.save_dir_GAN, 'simple_vocab.pkl'), 'w') as f:
		cPickle.dump((batcher.chars, batcher.vocab), f)

	logging.debug('Creating generator...')
	gan = GAN(args, is_training = True)
	 
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
		tf.initialize_all_variables().run()
		# Save only varaibles from gan
		saver = tf.train.Saver(tf.all_variables())

		ckpt = tf.train.get_checkpoint_state(args.save_dir_GAN)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		
		#################
		# Train generator
		#################
		for epoch in xrange(args.num_epochs_gen):
			# Anneal learning rate
			new_lr = args.learning_rate_gen * (args.decay_rate ** epoch)
			sess.run(tf.assign(gan.lr_gen, new_lr))
			batcher.reset_batch_pointer()
			state_gen = gan.initial_state_gen.eval()
			state_dis = gan.initial_state_dis.eval()

			for batch in xrange(batcher.num_batches):
				start = time.time()
				x, _  = batcher.next_batch()
				y     = np.ones(x.shape)
				feed  = {gan.input_data: x, 
						gan.targets: y, 
						gan.initial_state_gen: state_gen, 
						gan.initial_state_dis: state_dis}
				gen_train_loss, _ = sess.run([gan.gen_cost, gan.gen_train_op], feed)
				end   = time.time()

				print '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}' \
					.format(epoch * batcher.num_batches + batch,
						args.num_epochs * batcher.num_batches,
						epoch, gen_train_loss, end - start)
				
				if (epoch * batcher.num_batches + batch) % args.save_every == 0:
					checkpoint_path = os.path.join(args.save_dir_GAN, 'model.ckpt')
					saver.save(sess, checkpoint_path, global_step = epoch * batcher.num_batches + batch)
					print 'GAN model saved to {}'.format(checkpoint_path)
				

if __name__=='__main__':	
	args = parse_args()
	with tf.device('/gpu:3'):
		train_gan (args)



