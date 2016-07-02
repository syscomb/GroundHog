"""
Stochastic Gradient Descent.


TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"
import math
import numpy
import time
import logging

import theano
import theano.tensor as TT
from theano.scan_module import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.utils import print_time, print_mem, const

logger = logging.getLogger(__name__)

class SGD(object):
    def __init__(self,
                 model,
                 state,
                 data,
                 sampler):
        """
        Parameters:
            :param model:
                Class describing the model used. It should provide the
                 computational graph to evaluate the model, and have a
                 similar structure to classes on the models folder
            :param state:
                Dictionary containing the current state of your job. This
                includes configuration of the job, specifically the seed,
                the startign damping factor, batch size, etc. See main.py
                for details
            :param data:
                Class describing the dataset used by the model
        """

        if 'adarho' not in state:
            state['adarho'] = 0.96
        if 'adaeps' not in state:
            state['adaeps'] = 1e-6


        self.sampler = sampler
        #####################################
        # Step 0. Constructs shared variables
        #####################################
        bs = state['bs']
        self.model = model
        self.rng = numpy.random.RandomState(state['seed'])
        srng = RandomStreams(self.rng.randint(213))
        self.gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name)
                   for p in model.params]
        self.gnorm2 = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name+'_g2')
                   for p in model.params]
        self.dnorm2 = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name+'_d2')
                   for p in model.params]

        self.step = 0
        self.bs = bs
        self.state = state
        self.data = data
        self.step_timer = time.time()
        self.gdata = [theano.shared(numpy.zeros( (2,)*x.ndim,
                                                dtype=x.dtype),
                                    name=x.name) for x in model.inputs]

	if 'profile' not in self.state:
            self.state['profile'] = 0

        ###################################
        # Step 1. Compile training function
        ###################################
        logger.debug('Constructing grad function')
        loc_data = self.gdata
        self.prop_exprs = [x[1] for x in model.properties]
        self.prop_names = [x[0] for x in model.properties]
        self.update_rules = [x[1] for x in model.updates]
        rval = theano.clone(model.param_grads + self.update_rules + \
                            self.prop_exprs + [model.train_cost],
                            replace=zip(model.inputs, loc_data))
        nparams = len(model.params)
        nouts = len(self.prop_exprs)
        nrules = len(self.update_rules)
        gs = rval[:nparams]
        rules = rval[nparams:nparams + nrules]
        outs = rval[nparams + nrules:]

        norm_gs = TT.sqrt(sum(TT.sum(x**2)
            for x,p in zip(gs, self.model.params) if p not in self.model.exclude_params_for_norm))
        if 'cutoff' in state and state['cutoff'] > 0:
            c = numpy.float32(state['cutoff'])
            if state['cutoff_rescale_length']:
                c = c * TT.cast(loc_data[0].shape[0], 'float32')

            notfinite = TT.or_(TT.isnan(norm_gs), TT.isinf(norm_gs))
            _gs = []
            for g,p in zip(gs,self.model.params):
                if p not in self.model.exclude_params_for_norm:
                    tmpg = TT.switch(TT.ge(norm_gs, c), g*c/norm_gs, g)
                    _gs.append(
                       TT.switch(notfinite, numpy.float32(.1)*p, tmpg))
                else:
                    _gs.append(g)
            gs = _gs
        store_gs = [(s,g) for s,g in zip(self.gs, gs)]
        updates = store_gs + [(s[0], r) for s,r in zip(model.updates, rules)]

        rho = self.state['adarho']
        eps = self.state['adaeps']

        # grad2
        gnorm2_up = [rho * gn2 + (1. - rho) * (g ** 2.) for gn2,g in zip(self.gnorm2, gs)]
        updates = updates + zip(self.gnorm2, gnorm2_up)

        logger.debug('Compiling grad function')
        st = time.time()
        self.train_fn = theano.function(
            [], outs, name='train_function',
            updates = updates,
            givens = zip(model.inputs, loc_data),
            on_unused_input='warn')
        logger.debug('took {}'.format(time.time() - st))

        self.lr = numpy.float32(1.)
        new_params = [p - (TT.sqrt(dn2 + eps) / TT.sqrt(gn2 + eps)) * g
                for p, g, gn2, dn2 in
                zip(model.params, self.gs, self.gnorm2, self.dnorm2)]


        updates = zip(model.params, new_params)
        # d2
        d2_up = [(dn2, rho * dn2 + (1. - rho) *
            (((TT.sqrt(dn2 + eps) / TT.sqrt(gn2 + eps)) * g) ** 2.))
            for dn2, gn2, g in zip(self.dnorm2, self.gnorm2, self.gs)]
        updates = updates + d2_up

        self.update_fn = theano.function(
            [], [], name='update_function',
            allow_input_downcast=True,
            updates = updates)

        self.old_cost = 1e20
        self.schedules = model.get_schedules()
        self.return_names = self.prop_names + \
                ['cost',
                        'error',
                        'time_step',
                        'whole_time', 'lr']
        self.prev_batch = None

    def __call__(self):
        batch = self.data.next()
        assert batch

        # Perturb the data (! and the model)
        if isinstance(batch, dict):
            batch = self.model.perturb(**batch)
        else:
            batch = self.model.perturb(*batch)
        # Load the dataset into GPU
        # Note: not the most efficient approach in general, as it involves
        # each batch is copied individually on gpu

        sampleN = self.state['sampleN']

        myL = int(1.5*len(batch['y']))

        xi = []
        for i in xrange(self.state['num_systems']):
            xi.append(batch['x'+str(i)].squeeze())
        samples, probs = self.sampler(sampleN,myL,1,*xi)

        #samples, probs = self.sampler(sampleN,myL,1,batch['x'].squeeze())

        y,b = getUnique(samples, batch['y'], self.state)

        b = numpy.array(b,dtype='float32')
#        p = probs.sum(axis=0)
#        p = [math.exp(-i) for i in p]
#        p = [i/sum(p) for i in p]

#        print p
#        print b.mean()
#        print (b*p).mean()
        Y,YM = getYM(y, self.state)
#        print b
#        print Y
#        print YM

        diffN = len(b)

        for i in xrange(self.state['num_systems']):
            X = numpy.zeros((batch['x'+str(i)].shape[0], diffN), dtype='int64')
            batch['x'+str(i)] = batch['x'+str(i)]+X
            X = numpy.zeros((batch['x'+str(i)].shape[0], diffN), dtype='float32')
            batch['x_mask'+str(i)] = batch['x_mask'+str(i)]+X

        batch['y'] = Y
        batch['y_mask'] = YM
        batch['b'] = b

#        if not hasattr(self,'Last'):
#            self.Last = True
#            self.lastbatch = batch
#        else:
#            if self.Last:
#                batch = self.lastbatch
#                self.Last = False
#            else:
#                self.lastbatch = batch
#                self.Last = True
#        print batch['y']

        if isinstance(batch, dict):
            for gdata in self.gdata:
                gdata.set_value(batch[gdata.name], borrow=True)
        else:
            for gdata, data in zip(self.gdata, batch):
                gdata.set_value(data, borrow=True)
        # Run the trianing function
        g_st = time.time()
        rvals = self.train_fn()
        ############################################################
        #exported_grad = self.export_grad_fn()
        #print exported_grad
        ############################################################
        for schedule in self.schedules:
            schedule(self, rvals[-1])
        self.update_fn()
        g_ed = time.time()
        self.state['lr'] = float(self.lr)
        cost = rvals[-1]
        self.old_cost = cost
        whole_time = time.time() - self.step_timer
        if self.step % self.state['trainFreq'] == 0:
            msg = '.. iter %4d cost %.3f'
            vals = [self.step, cost]
            for dx, prop in enumerate(self.prop_names):
                msg += ' '+prop+' %.2e'
                vals += [float(numpy.array(rvals[dx]))]
            msg += ' step time %s whole time %s lr %.2e'
            vals += [print_time(g_ed - g_st),
                     print_time(time.time() - self.step_timer),
                     float(self.lr)]
            print msg % tuple(vals)
        self.step += 1
        ret = dict([('cost', float(cost)),
                    ('error', float(cost)),
                       ('lr', float(self.lr)),
                       ('time_step', float(g_ed - g_st)),
                       ('whole_time', float(whole_time))]+zip(self.prop_names, rvals))
        return ret



def getUnique(samples, y, state):
    dic = {}
    ty = y.squeeze().tolist()
    words = cutSen(ty, state)
    words = [str(i) for i in words]
    ref,lens = getRefDict(words)
    dic[' '.join(words)]=1.0
    
    n = len(samples[0])
    for i in range(n):
        sen = samples[:,i]
        sen = cutSen(sen.tolist(), state)
        words = [str(i) for i in sen]
        tmp = ' '.join(words)
        if tmp in dic:
            continue
        else:
            dic[tmp] = calBleu(words, ref, lens)
    l = []
    b = []
    for sen in dic:
        words = sen.split(' ')
        l.append(words)
        b.append(dic[sen])
    return l,b

def getYM(y,state):
    n = len(y)
    max = 0
    for i in range(n):
        tmp = len(y[i])
        if max < tmp:
            max = tmp

    Y = numpy.ones((max,n), dtype='int64')*state['null_sym_target']
    Ymask = numpy.zeros((max, n), dtype='float32')

    for i in range(n):
        si = y[i]
        ly = len(si)
        Y[0:ly,i] = y[i]
        Ymask[0:ly,i] = 1

    return Y, Ymask
        

def my_log(a):
    if a == 0:
        return -100000
    return math.log(a)
def cutSen(x,state):
    if state['null_sym_target'] not in x:
        return x
    else:
        return x[:x.index(state['null_sym_target'])+1]

def getRefDict(words):
    lens = len(words)
    now_ref_dict = {}
    for n in range(1,5):
        for start in range(lens-n+1):
            gram = ' '.join([str(p) for p in words[start:start+n]])
            if gram not in now_ref_dict:
                now_ref_dict[gram] = 1
            else:
                now_ref_dict[gram] += 1
    return now_ref_dict, lens

def calBleu(x,ref_dict,lens):

    length_trans = len(x)
    words = x
    closet_length = lens
    sent_dict = {}
    for n in range(1,5):
        for start in range(length_trans-n+1):
            gram = ' '.join([str(p) for p in words[start:start+n]])
            if gram not in sent_dict:
                sent_dict[gram] = 1
            else:
                sent_dict[gram] += 1
    correct_gram = [0,0,0,0]
    for gram in sent_dict:
        if gram in ref_dict:
            n = len(gram.split(' '))
            correct_gram[n-1] += min(ref_dict[gram], sent_dict[gram])
    bleu = [0.,0.,0.,0.]
    smooth = 0
    for j in range(4):
        if correct_gram[j] == 0:
            smooth = 1
    for j in range(4):
        if length_trans > j:
            bleu[j] = 1.*(correct_gram[j]+smooth)/(length_trans - j + smooth)
        else:
            bleu[j] = 1
    brev_penalty = 1
    if length_trans < closet_length:
        brev_penalty = math.exp(1 - closet_length*1./length_trans)
    now_bleu = brev_penalty*math.exp((my_log(bleu[0]) + my_log(bleu[1]) + my_log(bleu[2]) + my_log(bleu[3]))/4)
    return now_bleu


