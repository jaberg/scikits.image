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

:licence: modified BSD
"""
import copy
from collections import deque

import transform

class UndefinedValue(object):
    """
    The value of a Symbol's ``value`` attribute when no value is defined.
    """
class MissingValue(Exception):
    """
    A value required to evalute an expression is missing
    """

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

    def coerce(self, value):
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

    def make_constant(self, value):
        self.value = self.coerce(value)
        self.constant = True

    def values_eq(self, v0, v1, approx=False):
        """Return True iff v0 and v1 are [`approx`] equal in the context of this Type. 

        The return value when neither v0 nor v1 is conformant is undefined.
        
        """
        return v0 == v1

    def get_clear_value(self):
        """Return an object that represents an undefined value."""
        if self.constant:
            return self.value
        return UndefinedValue
    def clone(self):
        if self.constant:
            return self
        return copy.deepcopy(self)

class Symbol(object):
    """A value node in an expression graph.
    """
    Type=Type
    @classmethod
    def new(cls, closure=None, expr=None, type=None,value=UndefinedValue,name=None):
        """Return a Symbol instance within the given closure
        """
        if closure is None:
            closure = default_closure
        if type is None:
            type = cls.Type()
        if value is not type.get_clear_value():
            value = type.coerce(value)
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

class Closure(object):
    """A Closure encapsulates a set of Symbols that can be connected by Expressions.

    A closure makes it possible to monitor and coordinate changes to an expression graph.


    """
    def __init__(self):
        self.elements = set() # the Symbols in this Closure
        self.clone_dict = {}  # maps foreign Symbols to local ones.

    def clone(self, orig_symbol, recurse):
        """Return a copy of Symbol orig_symbol with the same attributes, but with self as closure """
        try:
            return self.clone_dict[orig_symbol]
        except KeyError:
            pass
        #print "CLONE", orig_symbol, orig_symbol.expr

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
            #print "CLONE returning", self.clone_dict[orig_symbol]
            return self.clone_dict[orig_symbol]
        else:
            # single clone of orig_symbol
            rval = self.add_symbol(
                    orig_symbol.__class__.new(
                        closure=self,
                        expr=None,
                        type=orig_symbol.type.clone(),
                        value=orig_symbol.value,
                        name=orig_symbol.name))
            self.clone_dict[orig_symbol] = rval
            #print "CLONE Returning", rval
            return rval

    def add_symbol(self, symbol):
        """Add a Symbol to this closure
        """
        if symbol.closure and symbol.closure is not self:
            raise ValueError('symbol already has closure', symbol.closure)
        self.elements.add(symbol)
        symbol.closure = self
        return symbol

    def add_expr(self, expr):
        """Add an expression to this closure"""
        for o in expr.outputs:
            o.expr = expr
        expr.closure = self
        return expr

    def remove_symbol(self, symbol):
        for s_foreign, s_local in self.clone_dict.items():
            if s_local is symbol:
                del self.clone_dict[s_foreign]
        self.elements.remove(symbol)
        symbol.closure = None

    def remove_expr(self, expr):
        for o in expr.outputs:
            o.expr = None
        o.closure = None

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
                    s.value = s.type.coerce(r)
                    if not (s.value is r):
                        print >> sys.stderr, "WARNING: %s returned non-conformant value" % str(expr.impl)
            else:
                # one output means `symbol` must be that one output
                symbol.value = symbol.type.coerce(results)
                if not (symbol.value is results):
                    print >> sys.stderr, "WARNING: %s returned non-conformant value" % str(expr.impl)
        return symbol.value

class SpecializedClosure(Closure):
    """A SpecializedClosure is a Closure that is specialized for computing specific output
    Symbols from specific input Symbols.


    Attributes:

     transform_policy       - a callable object that will rewire the graph [for faster
                              evaluation].
     on_is_valid            - set of callables that must all return True for the closure to be valid. They
                              are called with the closure as an argument.
     on_replace_symbol_pre  - set of callables run before replacing a symbol (closure, old_symbol, new_symbol)
     on_replace_symbol_post - set of callables run after replacing a symbol (closure, old_symbol, new_symbol)
     on_replace_symbols_pre - set of callables run before replacing a set of symbols (closure, old_new_list)
     on_replace_symbols_post- set of callables run after replacing a set of symbols (closure, old_new_list)
     on_replace_impl_pre    - set of callables run before replacing an impl (closure, expr, impl)
     on_replace_impl_post   - set of callables run after replacing an impl (closure, expr, impl)
     
    """
    def __init__(self, transform_policy):
        super(SpecializedClosure, self).__init__()
        self._iterating = False
        self._modified_since_iterating = False
        self.transform_policy = transform_policy
        self.revert_fns = []

        self.on_is_valid = set()
        self.on_replace_symbol_pre = set()
        self.on_replace_symbol_post = set()
        self.on_replace_symbols_pre = set()
        self.on_replace_symbols_post = set()
        self.on_replace_impl_pre = set()
        self.on_replace_impl_post = set()
    def add_symbol(self, symbol):
        rval = super(SpecializedClosure,self).add_symbol(symbol)
        if not hasattr(rval, 'clients'):
            rval.clients = set()
        return rval

    def add_expr(self, expr):
        """Add an expression to this closure"""
        rval = super(SpecializedClosure, self).add_expr(expr)
        for i, s_i in enumerate(expr.inputs):
            s_i.clients.add((expr, i))
        return rval
    def remove_expr(self, expr):
        rval = super(SpecializedClosure, self).remove_expr(expr)
        for i, s_i in enumerate(expr.inputs):
            s_i.clients.remove((expr, i))
        return rval

    def remove_symbol(self, symbol):
        assert symbol.closure == self
        if symbol in self.inputs:
            raise ValueError('Cannot remove input symbol', symbol)
        if symbol in self.outputs:
            raise ValueError('Cannot remove output symbol', symbol)
        if symbol.clients:
            raise ValueError('Cannot remove symbol with clients', symbol)
        super(SpecializedClosure, self).remove_symbol(symbol)
        if symbol.expr and all(((o.closure is None) and (not o.clients)) for o in symbol.expr.outputs):
            # no outputs are in use, so remove symbol.expr as client
            for pos, s_input in enumerate(symbol.expr.inputs):
                s_input.clients.remove((symbol.expr, pos))
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

    #
    # Methods for undoing transactions
    #

    def get_state(self):
        """
        Return a handle to the current state of the closure
        """
        return len(self.revert_fns)
    def revert_to_state(self, state):
        """
        Reset the state of the closure to what it was when `state` was returned by `get_state`.

        Invalidates states returned since `state`.
        """
        if state > len(self.revert_fns):
            raise ValueError('state has been invalidated by revert_to_state of a previous state')
        while len(self.revert_fns)>state:
            self.revert_fns.pop()()

    if 0:
        def add_alternative(self, symbol, new_symbol):
            """Register `new_symbol` as a potential replacement for symbol
            """
            #PRE-HOOK
            raise NotImplementedError()
            #POST-HOOK

    def replace_symbol(self, symbol, new_symbol):
        """
        Replace all references to symbol in closure with references to new_symbol.
        """
        for fn in self.on_replace_symbol_pre:
            fn(self, symbol, new_symbol)
        for client,position in symbol.clients:
            assert client.inputs[position] is symbol
            client.inputs[position] = new_symbol
        def undo():
            for client,position in symbol.clients:
                client.inputs[position] = symbol
        self.revert_fns.append(undo)
        if self._iterating:
            self._modified_since_iterating = True
        for fn in self.on_replace_symbol_post:
            fn(self, symbol, new_symbol)

    def replace_symbols(self, old_new_iterable):
        """
        Replace all symbols, or revert the transaction
        """
        old_new_iterable = list(old_new_iterable)
        for fn in self.on_replace_symbol_pre:
            fn(self, symbol, new_symbol)
        state = self.get_state()
        try:
            for s_old, s_new in old_new_iterable:
                self.replace_symbol(s_old, s_new)
        except:
            # in debugger: s_old and s_new were the troublemakers
            self.revert_to_state(state)
            raise

    def replace_impl(self, expr, new_impl):
        for fn in self.on_replace_impl_pre:
            fn(self, expr, new_impl)
        old_impl = expr.impl
        expr.impl = new_impl
        def undo():
            expr.impl = old_impl
        self.revert_fns.append(undo)
        if self._iterating:
            #most times replacing an implementation is fine, but sometimes
            # an implementation can have different view_map and destroy_map properties
            # which have an effect on the evaluation order.
            self._modified_since_iterating = True
        for fn in self.on_replace_impl_post:
            fn(self, expr, new_impl)

    def is_valid(self):
        """Return True IFF closure can be evaluated"""
        for fn in self.on_is_valid:
            if not fn(self):
                return False
        # sort the graph to make sure an order exists
        # TODO: consider making a plugin do this, so that 
        #       the plugin can be replaced by the destroyhandler
        io_toposort(self.inputs, self.outputs)
        return True
        
    def print_eval_order(self):
        for i,impl in enumerate(self.expr_iter()):
            print i,impl

    def nodes_iter(self, filter_fn):
        """Yield expr nodes in arbitrary order.

        Raises an exception if you try to continue iterating after
        modifying the expression graph.
        """
        things = [e for e in io_toposort(self.inputs, self.outputs, {}) if filter_fn(e)]
        self._iterating = True
        for e in things:
            if self._modified_since_iterating:
                raise Exception('Modified since iterating')
            assert e.closure == self
            yield e
        self._iterating = False
        self._modified_since_iterating = False

    def constant_iter(self):
        return self.nodes_iter(lambda o: hasattr(o, 'type') and getattr(o.type, 'constant', False))
    def symbol_iter(self):
        return self.nodes_iter(lambda o: hasattr(o, 'type'))
    def expr_iter(self):
        return self.nodes_iter(lambda o: hasattr(o, 'impl'))
    def expr_iter_with_Impl(self, ImplClass):
        return self.nodes_iter(lambda o: isinstance(getattr(o, 'impl', None), ImplClass))

    def __call__(self, *args):
        if len(args) != len(self.inputs):
            raise TypeError('Wrong number of inputs')
        computed = {}
        #print 'ALL', list(self.nodes_iter(lambda x:True))
        #print "ELEMENTS", list(self.elements)
        #print "CONSTANTS", list(self.constant_iter())
        #print "INPUTS", self.inputs
        #print "INPUTS0 CLIENTS", self.inputs[0].clients
        for c in self.constant_iter():
            computed[c] = c.value
        for i, a in zip(self.inputs, args):
            computed[i] = a
        print computed

        for expr in self.expr_iter():
            args = [computed[i] for i in expr.inputs]
            results = expr.impl.fn(*args)
            if expr.n_outputs>1:
                assert len(results) == len(expr.outputs)
                for s,r in zip(expr.outputs, results):
                    computed[s] = s.type.coerce(r) # coerce returns r (quickly) if r is conformant
                    if not (computed[s] is r):
                        print >> sys.stderr, "WARNING: %s returned non-conformant value" % str(expr.impl)
            else:
                # one output means `symbol` must be that one output
                computed[expr.outputs[0]] = expr.outputs[0].type.coerce(results)
                if not (computed[expr.outputs[0]] is results):
                    print >> sys.stderr, "WARNING: %s returned non-conformant value" % str(expr.impl)

        if self.unpack:
            return computed[self.outputs[0]]
        else:
            return [computed[o] for o in self.outputs]

# The default closure is a database of values to use for symbols
# that are not given as function arguments.
# The default closure is used by the compute() function to 
# support lazy evaluation.
default_closure = IncrementalClosure()

class Expr(object):
    """An implementation node in a expression graph. 
    
    This class is a glorified tuple, it has no interesting methods.
    """
    @classmethod
    def new(cls, impl, inputs, outputs):
        rval = cls(None, impl, inputs, outputs)
        #assert all Inputs and outputs are symbols
        bad_inputs = [i for i in inputs if not isinstance(i, Symbol)]
        bad_outputs =[i for i in outputs if not isinstance(i, Symbol)] 
        assert not bad_inputs, bad_inputs
        assert not bad_outputs, bad_outputs
        inputs[0].closure.add_expr(rval)
        return rval

    def __init__(self, closure, impl, inputs, outputs):
        self.closure = closure
        self.impl = impl
        self.inputs = list(inputs)
        self.outputs = list(outputs)
    def __str__(self):
        return 'Expr{%s}'%str(self.impl)
    n_inputs = property(lambda self: len(self.inputs))
    n_outputs = property(lambda self: len(self.outputs))

class Impl(object):
    """An implementation for an Expression in an expression graph.

    Attributes:
      fn - a normal [non-symbolic] function for which this Impl stands.
      n_outputs - the number of outputs fn will return
    """
    fn_protocol = 'normal' #alternatives are 'cond' and maybe others in future.
    view_map = {}
    destroy_map = {}
    n_outputs = 1
    name=None

    ExprCls=Expr

    @classmethod
    def allow_lazy(cls, *args, **kwargs):
        """
        A decorator for turning functions into the `fn` attribute of an Impl object.
        """
        def deco(fn):
            return cls(fn=fn,*args, **kwargs)
        return deco

    def __init__(self, fn, name=None):
        self.fn = fn
        self.name = name

    def __str__(self):
        return 'Impl_%s'%self.name

    @staticmethod
    def closure_from_args(args):
        """Return the closure if any of the args is Symbolic, else None"""
        for a in args:
            if isinstance(a, Symbol):
                return a.closure
    def new_expr(self, inputs, outputs):
        return self.ExprCls.new(self, inputs, outputs)

    def __call__(self, *args):
        """Return Symbolic output if any of `args` is a symbol, otherwise return `self.fn(*args)` """
        #
        # This method implements an argument un-packing heuristic (n_outputs)
        # This is supposed to be convenient most of the time.  If you are writing an Impl for
        # which the heuristic doesn't work, feel free to override __call__.
        #
        closure = self.closure_from_args(args)
        if closure:
            # return a symbolic result
            inputs = [self.as_input(closure, a) for a in args]
            outputs = self.outputs_from_inputs(inputs)
            expr = self.new_expr(inputs, outputs)
            if self.n_outputs>1:
                return outputs
            else:
                return outputs[0]
        else:
            return self.fn(*args)

    def infer_type(self, expr):
        """
        Update the meta-data of inputs and outputs.
        Return a set of types that were changed, or None if no changes were made.

        Raise TypeError() if inputs have become incompatible with expr.
        """
        pass

    def as_input(self, closure, obj, type_cls=Type):
        """Return a Symbol in `closure` for `obj`.

        This is a helper method used by Impl.__call__.

        If `obj` is not a symbol, it will be wrapped in one and assigned a type with the
        type_cls constructor.
        ``type = type_cls(constant=True, value=obj)``

        Raises ValueError on an obj of a foreign closure, does not catch any exceptions from
        type_cls.
        """
        if isinstance(obj, Symbol):
            if obj.closure is not closure:
                raise ValueError('Input in foreign closure', obj)
            return obj
        type = type_cls()
        type.make_constant(obj)
        rval = closure.add_symbol(Symbol.new(closure, type=type, value=type.value))
        return  rval

    def outputs_from_inputs(self, inputs):
        """Return `self.n_output` Symbols for the outputs of an Expr of this Impl.

        This is a helper method used by Impl.__call__.
        Override this method to return different types of symbols with Impl.__call__.
        """
        outputs = [Symbol.new(closure) for o in range(self.n_outputs)]

def default_function_closure_ctor():
    return SpecializedClosure(transform.TransformPolicy.new())

def function(inputs, outputs, 
        closure_ctor=default_function_closure_ctor,
        givens=None,
        updates=None):
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
    print closure.clone_dict
    cloned_inputs = [closure.clone(i, recurse=False) for i in inputs]
    cloned_outputs = [closure.clone(o, recurse=True) for o in outputs]
    closure.set_io(cloned_inputs, cloned_outputs, updates, return_outputs0)
    return closure

def io_toposort(inputs, outputs, orderings):
    """Returns sorted list of expression nodes

    inputs - list of inputs
    outputs - list of outputs
    orderings - dict of additions to the normal inputs and outputs constraints

    """

    assert isinstance(outputs, (tuple, list, deque))

    iset = set(inputs)

    expand_cache = {}
    start=deque(outputs)
    rval_set = set()
    rval_set.add(id(None))
    rval_list = list()
    expand_inv = {}
    sources = deque()
    while start:
        l = start.pop()# this makes the search dfs
        if id(l) not in rval_set:
            rval_list.append(l)
            rval_set.add(id(l))
            if l in iset:
                assert not orderings.get(l, [])
                expand_l = []
            else:
                try:
                    if l.expr:
                        expand_l = [l.expr]
                    else:
                        expand_l = []
                except AttributeError:
                    expand_l = list(l.inputs)
                expand_l.extend(orderings.get(l, []))
            if expand_l:
                for r in expand_l:
                    expand_inv.setdefault(r, []).append(l)
                start.extend(expand_l)
            else:
                sources.append(l)
            expand_cache[l] = expand_l
    assert len(rval_list) == len(rval_set)-1

    rset = set()
    rlist = []
    while sources:
        node = sources.popleft()
        if node not in rset:
            rlist.append(node)
            rset.add(node)
            for client in expand_inv.get(node, []):
                expand_cache[client] = [a for a in expand_cache[client] if a is not node]
                if not expand_cache[client]:
                    sources.append(client)

    if len(rlist) != len(rval_list):
        raise ValueError('graph contains cycles')
    return rlist
