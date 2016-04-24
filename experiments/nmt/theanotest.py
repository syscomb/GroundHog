import numpy
import theano
import theano.tensor as TT
from groundhog.layers import\
        Layer,\
        MultiLayer,\
        SoftmaxLayer,\
        HierarchicalSoftmaxLayer,\
        LSTMLayer, \
        RecurrentLayer,\
        RecursiveConvolutionalLayer,\
        UnaryOp,\
        Shift,\
        LastState,\
        DropOp,\
        Concatenate

fc0 = TT.tensor3()
bc0 = TT.tensor3()
fc1 = TT.tensor3()
bc1 = TT.tensor3()

c0 = Concatenate(axis=2)(*[fc0,bc0]).out
c1 = Concatenate(axis=2)(*[fc1,bc1]).out

c = Concatenate(axis=1)(*[c0,c1]).out

func = theano.function(inputs=[fc0, bc0,fc1,bc1], outputs=[c0,c1,c])

fc0v = numpy.asarray([[[.1,.1,.1],[.2,.2,.3]],[[.3,.3,.3],[.4,.4,.4]]], dtype=theano.config.floatX)
bc0v = numpy.asarray([[[.5,.5,.5],[.6,.6,.6]],[[.7,.8,.8],[.8,.8,.8]]], dtype=theano.config.floatX)
fc1v = numpy.asarray([[[1.1,1.1,1.1],[1.2,1.2,1.3]],[[1.3,1.3,1.3],[1.4,1.4,1.4]]], dtype=theano.config.floatX)
bc1v = numpy.asarray([[[1.5,1.5,1.5],[1.6,1.6,1.6]],[[1.7,1.8,1.8],[1.8,1.8,1.8]]], dtype=theano.config.floatX)

print func(fc0v,bc0v,fc1v,bc1v)