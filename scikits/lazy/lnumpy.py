"""
A numpy-like module with support for lazy evaluation

:licence: modified BSD
"""
import numpy
from .lazy import Symbol, Impl, Type

class NdarrayType(Type):
    # None means unknown
    # For things which are not generally true (symmetric, P.S.D)
    # it is supposed to be convenient that unknown and False are both negative in terms of an
    # if statement.  For boolean properties, make sure that False corresponds to the default
    # setting.
    constant = False
    value = None
    dtype = None
    shape = None
    strides=None
    databuffer=None
    contiguous=None
    symmetric=None
    positive_semidefinite=None
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def is_conformant(self, obj):
        if self.constant: return (self.value is obj)
        if type(obj) != numpy.ndarray: return False
        if (self.dtype != None) and obj.dtype != self.dtype: return False
        if self.shape is not None:
            if len(self.shape) != obj.ndim:
                return False
            if any(a!=b for (a,b) in zip(self.shape, obj.shape) if a is not None):
                return False
        if self.strides is not None:
            if len(self.strides) != obj.strides:
                return False
            if any(a!=b for (a,b) in zip(self.strides, obj.strides) if a is not None):
                return False
        if (self.databuffer != None) and obj.data != self.databuffer: return False
        return True
    def coerce(self, obj):
        """
        TODO: Explain how to think about whether this function
        could change `obj` or raise a TypeError.
        """
        if self.is_conformant(obj): return obj
        if self.constant: raise TypeError('', (obj, self.value))
        if self.databuffer is not None: raise TypeError()
        if self.dtype:
            nda = numpy.asarray(obj, self.dtype)
        else:
            nda = numpy.asarray(obj)
        if self.shape is not None:
            # left-padding the nda.shape with 1s is ok
            # left-trimming 1s from the nda.shape is ok
            # no other changes to nda.shape are ok
            if nda.ndim > len(self.shape):
                if all(nda.shape[:nda.ndim-len(self.shape)]==1):
                    nda.shape = nda.shape[nda.ndim-len(self.shape):]
            if nda.ndim < len(self.shape):
                nda.shape = (1,)*(len(self.shape) - nda.ndim) + nda.shape
            if len(self.shape) != nda.ndim:
                raise TypeError('rank mismatch', obj)
            if any(a!=b for (a,b) in zip(self.shape, nda.shape) if a is not None):
                raise TypeError('shape mismatch', obj)
        if self.strides is not None:
            if len(self.strides) != nda.ndim:
                raise TypeError('stride count mismatch', obj)
            if any(a!=b for (a,b) in zip(self.strides, nda.strides) if a is not None):
                raise TypeError('stride mismatch', obj)
        return nda
    def values_eq(self, v0, v1, approx=False):
        if approx:
            return numpy.allclose(v0, v1)
        else:
            return numpy.all(v0 == v1)
    def make_constant(self, value):
        super(NdarrayType, self).make_constant(value)
        self.dtype=self.value.dtype
        self.shape=self.value.shape
        self.strides=self.value.strides
        self.databuffer=self.value.data

    def __repr__(self):
        return 'NdarrayType{constant=%s,dtype=%s,shape=%s,strides=%s,contiguous=%s,symmetric=%s}'%(
            self.constant, self.dtype, self.shape, self.strides, self.contiguous, self.symmetric)


class NdarraySymbol(Symbol):
    Type = NdarrayType
    def __array__(self):
        return numpy.asarray(self.compute())

    def __add__(self, other): return add(self, other)
    def __sub__(self, other): return subtract(self, other)
    def __mul__(self, other): return multiply(self, other)
    def __div__(self, other): return divide(self, other)

    def __radd__(other, self): return add(self, other)
    def __rsub__(other, self): return subtract(self, other)
    def __rmul__(other, self): return multiply(self, other)
    def __rdiv__(other, self): return divide(self, other)

class NdarrayImpl(Impl):
    def outputs_from_inputs(self, inputs):
        closure = inputs[0].closure
        outputs = [NdarraySymbol.new(closure, type=NdarrayType.new()) for o in range(self.n_outputs)]
        return outputs

    def as_input(self, closure, obj, type_cls=NdarrayType):
        """Convenience method - it's the default constructor for lazy Impl __call__ methods to
        use to easily turn all inputs into symbols.
        """
        return super(NdarrayImpl, self).as_input(closure, obj, type_cls)

class Elemwise(NdarrayImpl):
    """
    Base for element-wise Implementations
    """
    def __str__(self):
        return 'NumPy_%s'%self.name

    def infer_type(self, expr):
        """Set output shapes according to numpy broadcasting rules
        """
        changed = set()
        super(Elemwise, self).infer_type(expr, changed)
        print 'INFER_TYPE', expr, expr.inputs
        # if all inputs have an ndarray type
        if all(isinstance(i.type, NdarrayType) for i in expr.inputs):
            # get the shapes of the inputs
            shapes = [i.type.shape for i in expr.inputs if i.type.shape is not None]
            print 'SHAPES', expr, shapes, expr.inputs
            # if all the inputs have a known number of dimensions
            if len(shapes) == len(expr.inputs):
                # the outputs has the rank of the highest-rank input
                out_shp = [None]*max(len(s) for s in shapes)
                # left-pad shapes that are too short
                shapes = [[1]*(len(out_shp)-len(s)) + list(s) for s in shapes]
                for dim in range(len(shapes)):
                    dim_known = not any(s[dim] is None for s in shapes)
                    if dim_known:
                        # TODO:
                        # could detect size errors here
                        out_shp[dim]= max(s[dim] for s in shapes)
                out_shp = tuple(out_shp)
                for o in expr.outputs:
                    if o.type.shape != out_shp:
                        o.type.shape = out_shp
                        changed.add(o)
        # INFER SYMMETRY
        if all([i.type.symmetric for i in expr.inputs]):
            for o in expr.outputs:
                if not o.type.symmetric:
                    o.type.symmetric = True
                    changed.add(o)
        return changed

class NumpyElemwise (Elemwise):
    """
    Base for element-wise Implementations with native numpy implementations
    """
    def __init__(self, name):
        super(NumpyElemwise, self).__init__(
                fn=getattr(numpy,name),
                name=name)

# Elementwise, Unary functions
# that upcast to float
class Elemwise_unary_float_upcast(NumpyElemwise):
    def __call__(self, x):
        return NumpyElemwise.__call__(self, x)
for name in ['exp', 'log', 'log2', 'log10', 'log1p',
        'tanh', 'cosh', 'sinh', 'tan', 'cos', 'sin']:
    globals()[name] = Elemwise_unary_float_upcast(name)

# Elementwise, binary functions
# that upcast to float
class Elemwise_binary_float_upcast(NumpyElemwise):
    def __call__(self, x, y):
        return NumpyElemwise.__call__(self, x, y)
for name in ['subtract', 'power', 'divide']:
    globals()[name] = Elemwise_binary_float_upcast(name)

# Elementwise, N-ary functions
# that upcast to float
class Elemwise_Nary_float_upcast(NumpyElemwise):
    pass
for name in ['add', 'multiply']:
    globals()[name] = Elemwise_Nary_float_upcast(name)

# Elementwise, range comparisons
class Elemwise_range_cmp(NumpyElemwise):
    def __call__(self, x, y):
        return NumpyElemwise.__call__(self, x, y)
for name in ['greater']:
    globals()[name] = Elemwise_range_cmp(name)


