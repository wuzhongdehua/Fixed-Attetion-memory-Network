import sys
import numpy as np
sys.path.append('../skip-thoughts')
import skipthoughts
from tqdm import tqdm
#from w2v_model import *
import hickle
import threading
import pickle
import tensorflow as tf
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import math
import sentence_cliper as sc
import data_augmentation
from six.moves import cPickle
model = None
#model = skipthoughts.load_model()

class Dataset(object):
    """This is a dataset ADT that contains story, QA.

    Args:
        param1 (dictionay) : (IMdb key, video clip name value) pair dictionary
        param2 (list) : QAInfo type list.

        We are able to get param1 and param2 by mqa.get_story_qa_data() function.
    """
    def __init__(self, nstory, story=None, qa=None):
        self.story = story
        self.qa = qa
        # embedding matrix Z = Word2Vec or Skipthoughts
        self.zq = [] # embedding matrix Z * questino q
        self.zsl = [] # embedding matrix Z * story sl
        self.zaj = [] # embedding matrix Z * answer aj
        self.ground_truth = [] # correct answer index
        self.zq_val = []
        self.zsl_val = []
        self.zaj_val = []
        self.ground_truth_val = []
        self.nstory = nstory
        self.index_in_epoch_train = 0
        self.index_in_epoch_val = 0
        self.num_train_examples = 0
        self.num_val_examples = 0

    def load_dataset(self, embedding_method, test_flag=False):
        if embedding_method == 'skip':
            skip_dim = 4800
            filename = '/data/movieQA/skip_split.hkl'
            skip_embed = hickle.load(filename)
            skip_embed['zsl_train'], skip_embed['zsl_val'] = sc.sent_clip(self.nstory, skip_embed, skip_dim)

            self.zq = skip_embed['zq_train']
            self.zsl = skip_embed['zsl_train']
            self.zaj = skip_embed['zaj_train']
            self.ground_truth = skip_embed['ground_truth_train']
            self.zq_val = skip_embed['zq_val']
            self.zsl_val = skip_embed['zsl_val']
            self.zaj_val = skip_embed['zaj_val']
            self.ground_truth_val = skip_embed['ground_truth_val']
            self.num_train_examples = self.zq.shape[0]
            self.num_val_examples = self.zq_val.shape[0]

        if embedding_method == 'word2vec':
            attention = hickle.load('/data/movieQA/hickle_dump/attention.hkl')
            w2v_embed = hickle.load('/data/movieQA/hickle_dump/w2v_concat.hkl')
            w2v_dim = 2500
            w2v_embed['zsl_train'], w2v_embed['zsl_val'], attention['train'], attention['val'] = sc.sent_clip(self.nstory, w2v_embed, attention, w2v_dim)

            self.zq = w2v_embed['zq_train']
            self.zsl = w2v_embed['zsl_train']
            self.zaj = w2v_embed['zaj_train']
            self.ground_truth = w2v_embed['ground_truth_train']
            self.attention = attention['train']
            self.zq_val = w2v_embed['zq_val']
            self.zsl_val = w2v_embed['zsl_val']
            self.zaj_val = w2v_embed['zaj_val']
            self.ground_truth_val = w2v_embed['ground_truth_val']
            self.attention_val = attention['val']

            print self.zq.shape
            print self.zsl.shape
            print self.zaj.shape
            print self.ground_truth.shape
            assert self.zq.shape == (9566, 1, w2v_dim)
            assert self.zsl.shape == (9566, self.nstory, w2v_dim)
            assert self.zaj.shape == (9566, 5, w2v_dim)
            assert self.ground_truth.shape == (9566, )
            assert self.attention.shape == (9566, self.nstory, 1)

            self.num_train_examples = self.zq.shape[0]
            self.num_val_examples = self.zq_val.shape[0]

    def next_batch(self, batch_size, type = 'train'):
        """ at training phase, getting training(or validation) data of predefined batch size.
        Args:
            param1 (int) : batch size
            param2 (string) : type of the data you want to get. You might choose between 'train' or 'val'

        Return:
            batch size of (zq, zaj, zsl, ground_truth) pair value would be returned.
        """

        if type == 'train':
            assert batch_size <= self.num_train_examples
            start = self.index_in_epoch_train
            self.index_in_epoch_train += batch_size
            if self.index_in_epoch_train > self.num_train_examples:
                """
                if batch index touch the # of exmaples,
                shuffle the training dataset and start next new batch
                """
                perm = np.arange(self.num_train_examples)
                np.random.shuffle(perm)
                self.zq = self.zq[perm]
                self.zsl = self.zsl[perm]
                self.ground_truth = self.ground_truth[perm]
                self.zaj = self.zaj[perm]
                self.attention = self.attention[perm]

                # start the next batch
                start = 0
                self.index_in_epoch_train = batch_size
            end = self.index_in_epoch_train
            return self.zsl[start:end], self.zq[start:end], self.zaj[start:end], self.ground_truth[start:end], self.attention[start:end]

        elif type == 'val':
            assert batch_size <= self.num_val_examples
            start = self.index_in_epoch_val
            self.index_in_epoch_val += batch_size
            if self.index_in_epoch_val > self.num_val_examples:
                perm = np.arange(self.num_val_examples)
                np.random.shuffle(perm)
                self.zq_val = self.zq_val[perm]
                self.zsl_val = self.zsl_val[perm]
                self.ground_truth_val = self.ground_truth_val[perm]
                self.zaj_val = self.zaj_val[perm]
                self.attention_val = self.attention_val[perm]

                start = 0
                self.index_in_epoch_val = batch_size
            end = self.index_in_epoch_val
            return self.zsl_val[start:end], self.zq_val[start:end], self.zaj_val[start:end], self.ground_truth_val[start:end], self.attention_val[start:end]
