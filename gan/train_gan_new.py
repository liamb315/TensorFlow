import tensorflow as tf
import numpy as np
import logging
from tensorflow.models.rnn import *
from argparse import ArgumentParser
from batcher_new import DiscriminatorBatcher, GANBatcher
from gan import GAN
from discriminator import Discriminator
import time
import os
import cPickle

logger = logging.getLogger()
logger.setLevel(logging.ERROR)


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--real_input_file', type=str, default='real_reviews.txt',
		help='real reviews')
	parser.add_argument('--fake_input_file', type=str, default='fake_reviews.txt',
		help='fake reviews')
	parser.add_argument('--data_dir', type=str, default='data/gan',
		help='data directory containing reviews')
	parser.add_argument('--vocab_file', type=str, default='simple_vocab.pkl',
		help='data directory containing reviews')
	parser.add_argument('--save_dir_GAN', type=str, default='models_GAN',
		help='directory to store checkpointed GAN models')
	parser.add_argument('--save_dir_dis', type=str, default='models_GAN/discriminator',
		help='directory to store checkpointed discriminator models')
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
	parser.add_argument('--num_epochs_GAN', type=int, default=5,
		help='number of epochs of GAN')
	parser.add_argument('--num_epochs_gen', type=int, default=1,
		help='number of epochs to train generator')
	parser.add_argument('--num_epochs_dis', type=int, default=1,
		help='number of epochs to train discriminator')
	parser.add_argument('--save_every', type=int, default=5,
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


def train_generator(gan, args, sess, initial_load = True):
	'''Train Generator via GAN'''
	logging.debug('Training generator...')
	
	batcher  = GANBatcher(args.real_input_file, args.vocab_file, 
						  args.data_dir, args.batch_size, 
						  args.seq_length)

	logging.debug('Vocabulary...')
	# TODO:  Why do this each time? Unnecessary
	with open(os.path.join(args.save_dir_GAN, 'config.pkl'), 'w') as f:
		cPickle.dump(args, f)
	with open(os.path.join(args.save_dir_GAN, 'simple_vocab.pkl'), 'w') as f:
		cPickle.dump((batcher.chars, batcher.vocab), f)

	# Save all GAN variables to gan_saver
	gan_vars = [v for v in tf.all_variables() if not 
					(v.name.startswith('classic/') or v.name.startswith('sampler/') )]
	gan_saver = tf.train.Saver(gan_vars)

	# Retrieve trainable Discriminator variables from dis_saver
	# TODO:  Trainable vs. All Variables?
	dis_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator/')]
	dis_saver = tf.train.Saver(dis_vars)

	if initial_load:
		logging.debug('Initla load of GAN parameters...')
		ckpt = tf.train.get_checkpoint_state(args.save_dir_GAN)
		if ckpt and ckpt.model_checkpoint_path:
			gan_saver.restore(sess, ckpt.model_checkpoint_path)
	
	else:
		logging.debug('Update GAN parameters from Discriminator...')		
		ckpt = tf.train.get_checkpoint_state(args.save_dir_dis)
		if ckpt and ckpt.model_checkpoint_path:
			dis_saver.restore(sess, ckpt.model_checkpoint_path)

	for epoch in xrange(args.num_epochs_gen):
		new_lr = args.learning_rate_gen * (args.decay_rate ** epoch)
		sess.run(tf.assign(gan.lr_gen, new_lr))
		batcher.reset_batch_pointer()
		state_gen = gan.initial_state_gen.eval()
		state_dis = gan.initial_state_dis.eval()

		for batch in xrange(10):
		# for batch in xrange(batcher.num_batches):
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
					args.num_epochs_gen * batcher.num_batches,
					epoch, gen_train_loss, end - start)
			
			if (epoch * batcher.num_batches + batch) % args.save_every == 0:
				checkpoint_path = os.path.join(args.save_dir_GAN, 'model.ckpt')
				gan_saver.save(sess, checkpoint_path, global_step = epoch * batcher.num_batches + batch)
				print 'GAN model saved to {}'.format(checkpoint_path)



def train_discriminator(discriminator, args, sess):
	'''Train the discriminator via classical approach'''
	logging.debug('Training discriminator...')
	
	batcher  = DiscriminatorBatcher(args.real_input_file, 
									args.fake_input_file, 
									args.data_dir, args.vocab_file,
									args.batch_size, args.seq_length)

	logging.debug('Vocabulary...')
	with open(os.path.join(args.save_dir_GAN, 'simple_vocab.pkl'), 'w') as f:
		cPickle.dump((batcher.chars, batcher.vocab), f)

	logging.debug('Loading GAN parameters to Discriminator...')
	# TODO:  Only using the trainable variables, is this OK?
	dis_vars = [v for v in tf.trainable_variables() if v.name.startswith('classic/')]
	dis_dict = {}
	for v in dis_vars:
		# Key:    op.name in GAN Checkpoint file
		# Value:  Local generator Variable 
		dis_dict[v.op.name.replace('classic/','discriminator/')] = v
	dis_saver = tf.train.Saver(dis_dict)

	ckpt = tf.train.get_checkpoint_state(args.save_dir_GAN)
	if ckpt and ckpt.model_checkpoint_path:
		dis_saver.restore(sess, ckpt.model_checkpoint_path)
	
	for epoch in xrange(args.num_epochs_dis):
		# Anneal learning rate
		new_lr = args.learning_rate_dis * (args.decay_rate ** epoch)
		sess.run(tf.assign(discriminator.lr, new_lr))
		batcher.reset_batch_pointer()
		state = discriminator.initial_state.eval()

		for batch in xrange(10):
		# for batch in xrange(batcher.num_batches):
			start = time.time()
			x, y  = batcher.next_batch()

			feed  = {discriminator.input_data: x, 
					 discriminator.targets: y, 
					 discriminator.initial_state: state}
			train_loss, state, _ = sess.run([discriminator.cost,
											discriminator.final_state,
											discriminator.train_op], 
											feed)
			end   = time.time()
			
			print '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}' \
				.format(epoch * batcher.num_batches + batch,
					args.num_epochs_dis * batcher.num_batches,
					epoch, train_loss, end - start)
			
			if (epoch * batcher.num_batches + batch) % args.save_every == 0:
				checkpoint_path = os.path.join(args.save_dir_dis, 'model.ckpt')
				dis_saver.save(sess, checkpoint_path, global_step = epoch * batcher.num_batches + batch)
				print 'Discriminator model saved to {}'.format(checkpoint_path)

			
def generate_samples(generator, args, sess, num_samples=1000):
	'''Generate samples from the current version of the GAN'''
	samples = []

	with open(os.path.join(args.save_dir_GAN, 'config.pkl')) as f:
		saved_args = cPickle.load(f)
	with open(os.path.join(args.save_dir_GAN, args.vocab_file)) as f:
		chars, vocab = cPickle.load(f)
	
	logging.debug('Loading GAN parameters to Generator...')
	gen_vars = [v for v in tf.all_variables() if v.name.startswith('sampler/')]
	gen_dict = {}
	for v in gen_vars:
		# Key:    op.name in GAN Checkpoint file
		# Value:  Local generator Variable 
		gen_dict[v.op.name.replace('sampler/','')] = v
	gen_saver = tf.train.Saver(gen_dict)
	ckpt = tf.train.get_checkpoint_state(args.save_dir_GAN)
	if ckpt and ckpt.model_checkpoint_path:
		gen_saver.restore(sess, ckpt.model_checkpoint_path)
	
	for _ in xrange(num_samples / args.batch_size):
		samples.append(generator.generate_samples(sess, saved_args, chars, vocab, args.n))

	return samples

def reset_reviews(data_dir, file_name):
	open(os.path.join(data_dir, file_name), 'w').close()


def adversarial_training(gan, discriminator, generator, args, sess):
	'''Adversarial Training'''
	train_generator(gan, args, sess, initial_load = True)
	generate_samples(generator, args, sess)

	for epoch in xrange(args.num_epochs_GAN):
		train_discriminator(discriminator, args, sess)
		train_generator(gan, args, sess, initial_load = False)
		reset_reviews(args.data_dir, args.fake_input_file)
		generate_samples(generator, args, sess, 100)
	return


if __name__=='__main__':
	args = parse_args()

	with tf.device('/gpu:3'):
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

			logging.debug('Creating models...')
			gan = GAN(args, is_training = True)
			with tf.variable_scope('classic'):
				discriminator = Discriminator(args, is_training = True)
			with tf.variable_scope('sampler'):
				generator = GAN(args, is_training = False)

			logging.debug('Initializing variables in graph...')
			tf.initialize_all_variables().run()

			adversarial_training(gan, discriminator, generator, args, sess)
