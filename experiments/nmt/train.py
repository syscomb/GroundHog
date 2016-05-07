#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint

import numpy

from groundhog.trainer.SGD_adadelta import SGD as SGD_adadelta
from groundhog.trainer.SGD import SGD as SGD
from groundhog.trainer.SGD_momentum import SGD as SGD_momentum
from groundhog.mainLoop import MainLoop
from experiments.nmt import\
        RNNEncoderDecoder, prototype_state, get_batch_iterator,get_batch_iterator_multi, SystemCombination
import experiments.nmt

logger = logging.getLogger(__name__)

class RandomSamplePrinter(object):

    def __init__(self, state, model, train_iter):
        args = dict(locals())
        args.pop('self')
        self.__dict__.update(**args)

    def __call__(self):
        def cut_eol(words):
            for i, word in enumerate(words):
                if words[i] == '<eol>':
                    return words[:i + 1]
            raise Exception("No end-of-line found")

        sample_idx = 0
        while sample_idx < self.state['n_examples']:
            batch = self.train_iter.next(peek=True)
            xs, ys = batch['x'], batch['y']
            for seq_idx in range(xs.shape[1]):
                if sample_idx == self.state['n_examples']:
                    break

                x, y = xs[:, seq_idx], ys[:, seq_idx]
                x_words = cut_eol(map(lambda w_idx : self.model.word_indxs_src[w_idx], x))
                y_words = cut_eol(map(lambda w_idx : self.model.word_indxs[w_idx], y))
                if len(x_words) == 0:
                    continue

                print "Input: {}".format(" ".join(x_words))
                print "Target: {}".format(" ".join(y_words))
                self.model.get_samples(self.state['seqlen'] + 1, self.state['n_samples'], x[:len(x_words)])
                sample_idx += 1

class RandomSamplePrinter_multi(object):

    def __init__(self, state, model, train_iter, enc_dec):
        args = dict(locals())
        args.pop('self')
        self.__dict__.update(**args)
        self.state = state
        self.enc_dec = enc_dec

    def __call__(self):
        def cut_eol(words):
            for i, word in enumerate(words):
                if words[i] == '<eol>':
                    return words[:i + 1]
            raise Exception("No end-of-line found")

        sample_idx = 0
        while sample_idx < self.state['n_examples']:
            batch = self.train_iter.next(peek=True)
            #print batch
            xs = []
            for i in xrange(self.state['num_systems']):
                xs.append(batch['x'+str(i)])
            ys = batch['y']
            for seq_idx in range(ys.shape[1]):
                if sample_idx == self.state['n_examples']:
                    break
                
                y = ys[:, seq_idx]
                y_words = cut_eol(map(lambda w_idx : self.model.word_indxs[w_idx], y))
                x = []
                x_words = []
                xpara = []
                for i in xrange(self.state['num_systems']):
                    x.append(xs[i][:, seq_idx])                
                    x_words.append(cut_eol(map(lambda w_idx : self.model.word_indxs_src[w_idx], x[i])))
                    print "Input: {}".format(" ".join(x_words[i]))
                    xpara.append(x[i][:len(x_words[i])])
                
                '''
                if len(x_words) == 0:
                    continue
                '''
                
                #print 'lenxpara:',len(xpara)
                #print 'xpara',xpara

                print "Target: {}".format(" ".join(y_words))
                #print self.enc_dec.sample_test()(*xpara)
                self.model.get_samples(self.state['seqlen'] + 1, self.state['n_samples'], *xpara)
                sample_idx += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--proto",  default="prototype_state",
        help="Prototype state to use for state")
    parser.add_argument("--skip-init", action="store_true",
        help="Skip parameter initilization")
    parser.add_argument("changes",  nargs="*", help="Changes to state", default="")
    return parser.parse_args()

def main():
    args = parse_args()
    print 'syscomb'

    state = getattr(experiments.nmt, args.proto)()
    if args.state:
        if args.state.endswith(".py"):
            state.update(eval(open(args.state).read()))
        else:
            with open(args.state) as src:
                state.update(cPickle.load(src))
    for change in args.changes:
        state.update(eval("dict({})".format(change)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    logger.debug("State:\n{}".format(pprint.pformat(state)))

    rng = numpy.random.RandomState(state['seed'])
    if state['syscomb']:
        enc_dec = SystemCombination(state, rng, args.skip_init)
    else:
        enc_dec = RNNEncoderDecoder(state, rng, args.skip_init)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    print 'lm model inputs:', lm_model.inputs

    logger.debug("Load data")
    if state['syscomb']:
        train_data = get_batch_iterator_multi(state)
        sampler = RandomSamplePrinter_multi(state, lm_model, train_data, enc_dec)
    else:
        train_data = get_batch_iterator(state)
        sampler = RandomSamplePrinter(state, lm_model, train_data)
    logger.debug("Compile trainer")
    algo = eval(state['algo'])(lm_model, state, train_data)
    logger.debug("Run training")
    main = MainLoop(train_data, None, None, lm_model, algo, state, None,
            reset=state['reset'],
            hooks=[sampler]
                if state['hookFreq'] >= 0
                else None)
    if state['reload']:
        main.load()
    if state['loopIters'] > 0:
        main.main()

if __name__ == "__main__":
    main()
