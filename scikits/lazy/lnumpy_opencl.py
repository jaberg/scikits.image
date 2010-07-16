"""
A OpenCL-based ndarray data type and optimizations to use it instead of lnumpy

:licence: modified BSD
"""
import sys
import numpy
from .lazy import Impl
from .lnumpy import NdarraySymbol as lnumpy_Symbol
from .lnumpy import NumpyElemwise

import pyopencl
from pyopencl.array import Array


class TransferToHost(Impl):
    def infer_type(self, expr):
        changed = set()
        i = expr.inputs[0]
        o = expr.outputs[0]
        for a in o.metadata_attributes():
            if getattr(o, a) != getattr(i, a):
                setattr(o, a, getattr(i, a))
                changed.add(o)
        return changed
    def outputs_from_inputs(self, inputs):
        return [lnumpy_Symbol.new(closure=inputs[0].closure)]


@TransferToHost.allow_lazy()
def transfer_to_host(device_array):
    return device_array.get()

class Symbol(lnumpy_Symbol):
    context = None
    def __NdarrayImpl_as_input__(self):
        return transfer_to_host(self)

    # The current implementation using pyopencl.array requires these
    # to be true.
    c_contiguous = property(lambda self: True)

    def clone(self, new_closure):
        # Cloning an OpenCL symbol
        # should use the same context as the original
        # Anyway, OpenCL contexts are not picklable
        context = self.context
        if self.context:
            delattr(self, 'context')
        rval = super(Symbol, self).clone(new_closure)
        rval.context = context
        return rval

    def is_conformant(self, obj):
        return super(Symbol, self).is_conformant(obj, type_required=Array)

    def coerce(self, obj, queue=None, async=False):
        if self.is_conformant(obj):
            return obj
        if isinstance(obj, Array):
            # TODO: handle a non-conforming Array without transfers
            nda = super(Symbol, self).coerce(obj.get())
        else:
            nda = super(Symbol, self).coerce(obj)
        #print 'coerce made obj', type(nda), nda.dtype, nda.shape, nda.strides

        assert isinstance(nda, numpy.ndarray)
        if queue is None:
            queue = pyopencl.CommandQueue(self.context)
        # async=False copying should mean that the queue is irrelevant for correctness,
        # although it might take a long time if the queue is already full of work.
        rval = pyopencl.array.to_device(self.context, queue, nda, async=async)
        return rval
