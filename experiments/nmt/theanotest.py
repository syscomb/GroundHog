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

c0 = Concatenate(axis=2)(*[fc0,bc0])

func = theano.function(inputs=[fc0, bc0], outputs=[c0])

fc0v = numpy.asarray([[[.1,.1,.1],[.2,.2,.3]],[[.3,.3,.3],[.4,.4,.4]]], dtype=theano.config.floatX)
bc0v = numpy.asarray([[[.5,.5,.5],[.6,.6,.6]],[[.7,.8,.8],[.8,.8,.8]]], dtype=theano.config.floatX)

print func(fc0v,bc0v)