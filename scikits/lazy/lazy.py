"""
Data structures to support lazy evaluation and lazy evaluation decorators.

The easiest way to support lazy evaluation of a function is to add a decorator:

.. code-block:: python

    @Impl.allow_lazy()
    def some_fn(*args):
        ...


In this case, if any argument to some_fn is a Symbol, then the output will also be a Symbol and
the function will not be run right away.  The input and output Symbols can also be used to
define specialized functions.

"""
import copy

import exprgraph
import transform

class UndefinedValue(object): pass
class MissingValue(Exception):pass

class Type(object):
    """
    Class to represent a set of possible data values.

    A Type is attached to a Symbol.

    Properties and attributes of a Type instance parametrize the kind of data that a Symbol
    might represent.

    """
    @classmethod
    def new(cls, constant=False, value=UndefinedValue, *args, **kwargs):
        return cls(*args, **kwargs)
    constant = False
    value = UndefinedValue
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def is_conformant(self, value):
        """
        Return True iff value is consistent with this Type.

        This function can be used in self-verifying execution modes,
        to ensure that Impls return the sort of objects they promised to return.
        """
        if self.constant:
            return (self.value is value)
        return True

    def make_conformant(self, value):
        """
        Return an object that is equal to value, but conformant to this metadata.

        This may involve casting integers to floating-point numbers, 
        padding with broadcastable dimensions, changing string encodings, etc.

        This method raises TypeError if `value` cannot be made conformant.
        """
        if self.is_conformant(value):
            return value
        if self.constant:
            raise TypeError(value)
        return value

    def eq(self, v0, v1, approx=False):
        """Return True iff v0 and v1 are [`approx`] equal in the context of this Type. 

        The return value when neither v0 nor v1 is conformant is undefined.
        
        """
        return v0 == v1

    def get_clear_value(self):
        """Return an object that represents an undefined value."""
        if self.constant:
            return self.value
        return UndefinedValue

class Symbol(object):
    """A value node in an expression graph.
    """
    Type=Type
    @classmethod
    def new(cls, closure, expr=None, type=None,value=UndefinedValue,name=None):
        """Return a Symbol instance within the given closure
        """
        if type is None:
            type = cls.Type()
        if value is not type.get_clear_value():
            value = type.make_conformant(value)
        rval = cls(None, expr,type,value,name)
        return closure.add_symbol(rval)

    def __init__(self, closure, expr, type, value, name):
        self.closure = closure
        self.expr = expr
        self.type = type
        self.value = value
        self.name = name

    def compute(self):
        """Return the value for this variable
        
        May not be supported by all closures, may raise errors on undefined inputs, but might
        work.  At least the IncrementalClosure (the one that you mainly interact with) supports
        this method.
        """
        return self.closure.compute_value(self)

exprgraph.Symbol=Symbol # hack around circular dependency
class Closure(object):
    """A Closure encapsulates a set of Symbols that can be connected by Expressions.

    A closure makes it possible to monitor and coordinate changes to an expression graph.


    """
    def __init__(self ):
        self.elements = set() # the Symbols in this Closure
        self.clone_dict = {}  # maps foreign Symbols to local ones.

    def clone(self, orig_symbol, recurse):
        """Return a copy of Symbol orig_symbol with the same attributes, but with self as closure """
        try:
            return self.clone_dict[orig_symbol]
        except KeyError:
            pass

        if getattr(orig_symbol, 'expr', None) and recurse:
            # recursive clone of inputs
            input_clones=[self.clone(i_s, recurse) for i_s in orig_symbol.expr.inputs]
            if orig_symbol.expr.n_outputs == 1:
                rval = orig_symbol.expr.impl(*input_clones)
                self.clone_dict[orig_symbol] = rval
            else:
                rval = orig_symbol.expr.impl(*input_copies)
                for s,r in zip(orig_symbol.expr.outputs, rval):
                    self.clone_dict[s] = r
            return self.clone_dict[orig_symbol]
        else:
            # single clone of orig_symbol
            return self.add_symbol(
                    orig_symbol.__class__.new(
                        closure=self,
                        expr=None,
                        type=copy.deepcopy(orig_symbol.type),
                        value=orig_symbol.value,
                        name=orig_symbol.name))

    def add_symbol(self, symbol):
        """Add a Symbol to this closure
        """
        if not isinstance(symbol, Symbol):
            raise TypeError(symbol)
        if symbol.closure and symbol.closure is not self:
            raise ValueError('symbol already has closure', symbol.closure)
        self.elements.add(symbol)
        symbol.closure = self
        return symbol

class IncrementalClosure(Closure):
    """
    A Closure that is designed for one-time incremental evaluation.

    This is the closure that is appropriate for lazy evaluation.
    It doesn't spend time on any transformations or compilation.

    This class implements the default_closure that the lazy decorators use to store Symbolic inputs.

    This class stores computed values into the Symbol.value attribute.

    .. TODO: How to permit garbage collection of Symbols that are *only* referenced by this
        closure?

    """
    def __init__(self):
        super(IncrementalClosure, self).__init__()
    def compute_value(self, symbol):
        if symbol.value is UndefinedValue:
            expr = symbol.expr
            if expr is None:
                raise MissingValue(symbol)
            args = [self.compute_value(i) for i in expr.inputs]
            results = expr.impl.fn(*args)
            if expr.n_outputs>1:
                for s,r in zip(expr.outputs, results):
                    s.value = s.type.make_conformant(r)
                    if not (s.value is r):
                        print >> sys.stderr, "WARNING: %s returned non-conformant value" % str(expr.impl)
            else:
                # one output means `symbol` must be that one output
                symbol.value = symbol.type.make_conformant(results)
                if not (symbol.value is results):
                    print >> sys.stderr, "WARNING: %s returned non-conformant value" % str(expr.impl)
        return symbol.value

class SpecializedClosure(Closure):
    """A SpecializedClosure is a Closure that is specialized for computing specific output
    Symbols from specific input Symbols.


    Attributes:

     transform_policy - a callable object that will rewire the graph [for faster
                        evaluation].
     
    """
    def __init__(self, transform_policy):
        super(CallableClosure, self).__init__()
        self._iterating = False
        self._modified_since_iterating = False
        self.transform_policy = transform_policy


    def set_io(self, inputs, outputs, updates, unpack_single_output):
        if updates:
            #TODO: translate the updates into the cloned graph
            raise NotImplementedError('updates arg is not implemented yet')
        for o in inputs+outputs:
            if o.closure is not self:
                raise ValueError('output not in closure', o)
        self.inputs = inputs
        self.outputs = outputs
        self.unpack = unpack_single_output and len(outputs)==1
        self.transform_policy(self)

    def change_input(self, expr, position, new_symbol):
        """Change
        """
        #PRE-HOOK
        raise NotImplementedError()
        #POST-HOOK

        #TODO: install a change_input post-hook to set modified_since_iterating to True
        # call pre-hooks
        if self._iterating:
            self._modified_since_iterating = True
        # call post-hooks
    def replace_impl(self, expr, new_impl):
        #PRE-HOOK
        expr.impl = new_impl
        #POST-HOOK

    def print_eval_order(self):
        for i,impl in enumerate(self.expr_iter()):
            print i,impl

    def nodes_iter(self, filter_fn):
        """Yield expr nodes in arbitrary order.

        Raises an exception if you try to continue iterating after
        modifying the expression graph.
        """
        things = [e for e in exprgraph.io_toposort(self.inputs, self.outputs) if filter_fn(e)]
        self._iterating = True
        for e in things:
            if self._modified_since_iterating:
                raise Exception('Modified since iterating')
            yield e
        self._iterating = False
        self._modified_since_iterating = False

    def constant_iter(self):
        return self.nodes_iter(lambda o: hasattr(o, 'type') and getattr(o.type, 'constant', False))
    def symbol_iter(self):
        return self.nodes_iter(lambda o: hasattr(o, 'type'))
    def expr_iter(self):
        return self.nodes_iter(lambda o: hasattr(o, 'impl'))

    def __call__(self, *args):
        if len(args) != len(self.inputs):
            raise TypeError('Wrong number of inputs')
        for i, a in zip(self.inputs, args):
            self.set_value(i,a)
        if self.unpack:
            return self.compute_value(self.outputs[0])
        else:
            return self.compute_values(self.outputs)

# The default closure is a database of values to use for symbols
# that are not given as function arguments.
# The default closure is used by the compute() function to 
# support lazy evaluation.
default_closure = IncrementalClosure()

class Impl(object):
    """

    Attributes:
      fn - a normal [non-symbolic] function that does the computations
    """
    @staticmethod
    def allow_lazy(*args, **kwargs):
        def deco(fn):
            return Impl(fn=fn,*args, **kwargs)
        return deco

    def __init__(self, fn, n_outputs=1, name=None, closure=default_closure):
        self.n_outputs=n_outputs
        self.fn = fn
        self.name = name
        self.closure = closure

    def __str__(self):
        return 'Impl_%s'%self.name

    def Expr(self, args, outputs):
        rval = Expr(self, args, outputs)
        for o in rval.outputs:
            o.expr = rval
        return rval

    @staticmethod
    def closure_from_args(args):
        for a in args:
            if isinstance(a, Symbol):
                return a.closure

    def __call__(self, *args):
        closure = self.closure_from_args(args)

        if closure:
            inputs = [self.as_input(closure, a) for a in args]
            outputs = self.outputs_from_inputs(inputs)
            expr = self.Expr(inputs, outputs)
            if self.n_outputs>1:
                return outputs
            else:
                return outputs[0]
        else:
            return self.fn(*args)

    def infer_type(self, expr, changed):
        """
        Update the meta-data of inputs and outputs.

        Explicitly mark .meta attributes as being modified by setting
        <symbol>.meta.changed = True
        """
        pass

    def as_input(self, closure, obj, constant_type=Constant):
        """Convenience method - it's the default constructor for lazy Impl __call__ methods to
        use to easily turn all inputs into symbols.
        """
        if isinstance(obj, Symbol):
            if obj.closure is not closure:
                raise ValueError('Input in foreign closure', obj)
            return obj
        rval = closure.add_symbol(Symbol.new(closure, type=constant_type(obj)))
        closure.set_value(rval, rval.type.value)
        return  rval


    def outputs_from_inputs(self, inputs):
        outputs = [Symbol.new(closure) for o in range(self.n_outputs)]

class Expr(object):
    """An implementation node in a expression graph.  """
    def __init__(self, impl, inputs, outputs):
        self.impl = impl
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        #assert all Inputs and outputs are symbols
        bad_inputs = [i for i in inputs if not isinstance(i, Symbol)]
        bad_outputs =[i for i in outputs if not isinstance(i, Symbol)] 
        assert not bad_inputs, bad_inputs
        assert not bad_outputs, bad_outputs
    def __str__(self):
        return 'Expr{%s}'%str(self.impl)
    n_inputs = property(lambda self: len(self.inputs))
    n_outputs = property(lambda self: len(self.outputs))
exprgraph.Expr=Expr

def default_function_closure_ctor():
    return SpecializedClosure(transform.TransformPolicy.new())
def function(inputs, outputs, closure_ctor=default_function_closure_ctor,
        givens=None, updates=None):
    if isinstance(outputs, Symbol):
        outputs = [outputs]
        return_outputs0 = True
    else:
        return_outputs0 = False

    if givens:
        #TODO: use the givens to modify the clone operation
        raise NotImplementedError('givens arg is not implemented yet')
    if updates:
        #TODO: clone the updates
        raise NotImplementedError('updates arg is not implemented yet')

    closure = closure_ctor()
    cloned_inputs = [closure.clone(i) for i in inputs]
    replacements = dict(zip(inputs, cloned_inputs))
    cloned_outputs = [closure.clone(o, replacements) for o in outputs]
    closure.set_io(cloned_inputs, cloned_outputs, updates, return_outputs0)

    cloned_inputs = [closure.add_symbol(i.clone(closure,as_constant=False)) for i in inputs]
    return closure

