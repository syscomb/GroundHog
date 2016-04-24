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

fc0 = TT.matrix()
bc0 = TT.matrix()
fc1 = TT.matrix()
bc1 = TT.matrix()

c0 = Concatenate(axis=2)(fc0,bc0)

func = theano.function(inputs=[fc0, bc0], outputs=[c0])

fc0v = numpy.asarray([[.2,.4,.6],[.1,.3,.5]], dtype=theano.config.floatX)
bc0v = numpy.asarray([[.4,.2,.6],[.7,.3,.5]], dtype=theano.config.floatX)

print func(fc0v,bc0v)