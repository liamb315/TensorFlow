import numpy as np
import os
import collections
import cPickle
import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Batcher(object):
	def __init__(self, data_dir, batch_size, seq_length):
		self.batch_size = batch_size
		self.seq_length = seq_length

		input_file  = os.path.join(data_dir, 'real_beer_reviews.txt')
		vocab_file  = os.path.join(data_dir, 'real_beer_vocab.pkl')
		tensor_file = os.path.join(data_dir, 'real_beer_data.npy')

		if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
			self.preprocess(input_file, vocab_file, tensor_file)
		else:
			self.load_preprocessed(vocab_file, tensor_file)

		self.create_batches()
		self.reset_batch_pointer()

	def preprocess(self, input_file, vocab_file, tensor_file):
		logging.debug('Reading text file...')
		with open(input_file, 'r') as f:
			data = f.read()
		counter         = collections.Counter(data)
		count_pairs     = sorted(counter.items(), key=lambda x: -x[1])
		self.chars, _   = list(zip(*count_pairs))
		self.vocab_size = len(self.chars)
		self.vocab      = dict(zip(self.chars, range(len(self.chars))))
		with open(vocab_file, 'w') as f:
			cPickle.dump(self.chars, f)
		self.tensor     = np.array(map(self.vocab.get, data))
		np.save(tensor_file, self.tensor)

	def load_preprocessed(self, vocab_file, tensor_file):
		logging.debug('Loading preprocessed files...')
		with open(vocab_file, 'r') as f:
			self.chars = cPickle.load(f)
		self.vocab_size  = len(self.chars)
		self.vocab       = dict(zip(self.chars, range(len(self.chars))))
		self.tensor      = np.load(tensor_file)

	def create_batches(self):
		logging.debug('Creating batches...')
		self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)
		self.tensor      = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
		x_data           = self.tensor
		y_data           = np.copy(self.tensor)	
		y_data[:-1]      = x_data[1:] # Labels are simply the next char
		y_data[-1]       = x_data[0]
		self.x_batches   = np.split(x_data.reshape(self.batch_size, -1), self.num_batches, 1)
		self.y_batches   = np.split(y_data.reshape(self.batch_size, -1), self.num_batches, 1)

	def next_batch(self):
		x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
		self.pointer += 1 
		return x, y

	def reset_batch_pointer(self):
		self.pointer = 0