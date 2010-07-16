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
import copy, sys
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

class Symbol(object):
    """A value node in an expression graph.

    Attributes
      value - this symbol stands for this particular value
      name - for debugging and pretty-printing
      expr - the Expr of which this is an output (may be None)
      clients - set of (expr, position) pairs where this symbol is an input (may be absent)
      closure - the Closure in which this symbol lives
    """
    constant = property(lambda s: s.value is not UndefinedValue)
    @classmethod
    def new(cls, closure=None, expr=None, value=UndefinedValue,name=None):
        """Return a Symbol instance within the given closure
        """
        if closure is None:
            closure = default_closure
        rval = cls(set(),None, expr,value,name)
        if value is not UndefinedValue:
            rval.update_from_value()
        return closure.add_symbol(rval)

    @classmethod
    def new_kwargs(cls, closure=None, expr=None, value=UndefinedValue, name=None, **kwargs):
        """ Return a new instance of this class with attributes initialized with kwargs
        """
        rval = cls.new(closure, expr, value, name)
        for key, val in kwargs.items():
            setattr(rval, key, val)
        return rval

    def __init__(self, clients, closure, expr, value, name):
        self.clients = clients
        self.closure = closure
        self.expr = expr
        self.value = value
        self.name = name

    def update_from_value(self):
        """Considering self.value to be permanent, update any metadata attributes"""
        pass

    def is_conformant(self, value):
        """
        Return True iff value is consistent with this Symbol.

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

    def clone(self, new_closure):
        if self.value is UndefinedValue:
            # basically do a deepcopy, but treat a few fields specially:
            dct = dict(self.__dict__)
            del dct['closure']
            del dct['expr']
            if 'clients' in dct: del dct['clients']
            rval = self.__class__.new(closure=new_closure, expr=None)
            rval.__dict__.update(copy.deepcopy(dct))
            return rval
        else:
            return self.__class__.new(closure=new_closure, value=self.value)

    def values_eq(self, v0, v1, approx=False):
        """Return True iff v0 and v1 are [`approx`] equal in the context of this Type. 

        The return value when neither v0 nor v1 is conformant is undefined.
        
        """
        return v0 == v1


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
            rval = orig_symbol.clone(self)
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

    def __getitem__(self, key):
        return self.clone_dict[key]

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
                    s.value = s.coerce(r)
                    if not (s.value is r):
                        print >> sys.stderr, "WARNING: %s returned non-conformant value" % str(expr.impl)
            else:
                # one output means `symbol` must be that one output
                symbol.value = symbol.coerce(results)
                if not (symbol.value is results):
                    print >> sys.stderr, "WARNING: %s returned non-conformant value" % str(expr.impl)
        return symbol.value

class FunctionClosure(Closure):
    """A FunctionClosure is a Closure that is specialized for computing specific output
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

     inputs  - list of symbols for which call arguments  provide values
     outputs - list of symbols to return from call
     
    """
    @classmethod
    def new(cls, inputs, outputs,
            givens=None, updates=None,
            transform_policy_ctor=transform.TransformPolicy.new):
        """Return a FunctionClosure cloned from `inputs` and `outputs`, optimized for computing
        `outputs` from `inputs.

        :type inputs: list of symbols
        :param inputs: 
            symbols standing for the parameter list of the function this closure represents.
            This constructor recognizes boolean Symbol attributes `mutable` and `strict`.  If
            an input symbol is `strict` then arguments will not be coerced to conform to the
            requirements of the symbol if they do not meet the conformation requirements of the
            symbol.  If an input symbol is `mutable` then the __call__ method of the closure
            may overwrite the contents of an argument to improve performance.

        :type givens: 
            iterable over pairs (Var1, Var2) of Variables. List, tuple or dict.

        :param givens: 
            substitutions to make in the computation graph (Var2 replaces
            Var1).  If `givens` is a list or tuple, then the substitutions are done in the
            order of the list.  If `givens` is a dictionary, then the order is undefined.

        """
        if isinstance(outputs, Symbol):
            outputs = [outputs]
            return_outputs0 = True
        else:
            return_outputs0 = False

        rval = cls(transform_policy_ctor())

        if givens:
            # initialize the clone_d mapping with the `givens` argument
            try:  # try to convert a dictionary to the sort of list that we need.
                givens = givens.items() 
            except:
                pass
            for s_orig, s_repl in givens:
                if not isinstance(s_orig, Symbol):
                    raise TypeError('given keys must be Symbol', s_orig)
                if not isinstance(s_repl, Symbol):
                    #TODO: some auto-symbol maker factory function or smth
                    raise NotImplementedError('given values must be Symbols for now', 
                            s_repl)
                if s_orig in rval.clone_dict:
                    #TODO: better type of exception here?
                    raise ValueError(
                            'Error in givens: symbol already cloned',
                            (s_orig, rval.clone_dict[s_orig]))
                cloned_repl = rval.clone(s_repl, recurse=True)
                assert s_orig not in rval.clone_dict
                rval.clone_dict[s_orig] = cloned_repl
        if updates:
            #TODO: clone the updates
            raise NotImplementedError('updates arg is not implemented yet')

        #rval.clone(i) will retrieve the cloned_repl if i was in givens
        cloned_inputs = [rval.clone(i, recurse=False) for i in inputs]
        cloned_outputs = [rval.clone(o, recurse=True) for o in outputs]
        rval.set_io(cloned_inputs, cloned_outputs, updates, return_outputs0)
        return rval

    reuse_computed = False
    def __init__(self, transform_policy):
        super(FunctionClosure, self).__init__()
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
        rval = super(FunctionClosure,self).add_symbol(symbol)
        if not hasattr(rval, 'clients'):
            rval.clients = set()
        return rval
    def remove_symbol(self, symbol):
        assert symbol.closure == self
        if symbol in self.inputs:
            raise ValueError('Cannot remove input symbol', symbol)
        if symbol in self.outputs:
            raise ValueError('Cannot remove output symbol', symbol)
        if symbol.clients:
            raise ValueError('Cannot remove symbol with clients', symbol)
        super(FunctionClosure, self).remove_symbol(symbol)
        if symbol.expr and all(((o.closure is None) and (not o.clients)) for o in symbol.expr.outputs):
            # no outputs are in use, so remove symbol.expr as client
            for pos, s_input in enumerate(symbol.expr.inputs):
                s_input.clients.remove((symbol.expr, pos))
    def replace_symbol(self, symbol, new_symbol):
        """
        Replace all references to symbol in closure with references to new_symbol.
        """
        for fn in self.on_replace_symbol_pre:
            fn(self, symbol, new_symbol)
        for client,position in symbol.clients:
            assert client.inputs[position] is symbol
            client.inputs[position] = new_symbol
        self.inputs = [(new_symbol if s is symbol else s) for s in self.inputs]
        self.outputs = [(new_symbol if s is symbol else s) for s in self.outputs]
        def undo():
            for client,position in symbol.clients:
                client.inputs[position] = symbol
            self.inputs = [(symbol if s is new_symbol else s) for s in self.inputs]
            self.outputs = [(symbol if s is new_symbol else s) for s in self.outputs]
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



    def add_expr(self, expr):
        """Add an expression to this closure"""
        rval = super(FunctionClosure, self).add_expr(expr)
        assert rval is expr
        for i, s_i in enumerate(expr.inputs):
            s_i.clients.add((expr, i))
        return rval
    def remove_expr(self, expr):
        rval = super(FunctionClosure, self).remove_expr(expr)
        for i, s_i in enumerate(expr.inputs):
            s_i.clients.remove((expr, i))
        return rval
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


    def set_io(self, inputs, outputs, updates, unpack_single_output):
        if updates:
            #TODO: translate the updates into the cloned graph
            raise NotImplementedError('updates arg is not implemented yet')
        for o in inputs+outputs:
            if o.closure is not self:
                raise ValueError('output not in closure', o)
        self.inputs_strict = [getattr(i, 'strict', False) for i in inputs]
        self.n_inputs = len(inputs)   # constant do not change!
        self.n_outputs = len(outputs) # constant do not change!
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
        return self.nodes_iter(lambda o: getattr(o, 'constant', False))
    def symbol_iter(self):
        return self.nodes_iter(lambda o: hasattr(o, 'expr'))
    def expr_iter(self):
        return self.nodes_iter(lambda o: hasattr(o, 'impl'))
    def expr_iter_with_Impl(self, ImplClass):
        return self.nodes_iter(lambda o: isinstance(getattr(o, 'impl', None), ImplClass))

    def __call__(self, *args, **kwargs):
        verbose = kwargs.get('verbose', False)
        if len(args) != len(self.inputs):
            raise TypeError('Wrong number of inputs')
        assert len(self.inputs) == self.n_inputs == len(self.inputs_strict)
        if self.reuse_computed:
            computed = getattr(self, 'computed', {})
        else:
            computed = {}
        #print 'ALL', list(self.nodes_iter(lambda x:True))
        #print "ELEMENTS", list(self.elements)
        #print "CONSTANTS", list(self.constant_iter())
        #print "INPUTS", self.inputs
        #print "INPUTS0 CLIENTS", self.inputs[0].clients
        if not computed:
            for c in self.constant_iter():
                computed[c] = c.value
        for strict, i, a in zip(self.inputs_strict, self.inputs, args):
            if strict and not i.is_conformant(a):
                raise TypeError(a)
            #print 'CALL: loading input', i, a.dtype, a.strides
            computed[i] = i.coerce(a)

        for expr in self.expr_iter():
            args = [computed[i] for i in expr.inputs]
            if expr.impl.fn_protocol == 'python':
                results = expr.impl.fn(*args)
            elif expr.impl.fn_protocol == 'python_io':
                old_rvals = [computed.get(i,UndefinedValue) for i in expr.outputs]
                if verbose:
                    print >> sys.stderr, "FunctionClosure::__call__ eval", expr
                results = expr.impl.fn(args, old_rvals)
            else:
                raise NotImplementedError('Impl.fn_protocol', expr.impl.fn_protocol)
            if expr.n_outputs>1:
                assert len(results) == len(expr.outputs)
                for s,r in zip(expr.outputs, results):
                    computed[s] = s.coerce(r) # coerce returns r (quickly) if r is conformant
                    if not (computed[s] is r):
                        print >> sys.stderr, "WARNING: %s returned non-conformant value" % str(expr.impl)
            else:
                # one output means `symbol` must be that one output
                computed[expr.outputs[0]] = expr.outputs[0].coerce(results)
                if not (computed[expr.outputs[0]] is results):
                    print >> sys.stderr, "WARNING: %s returned non-conformant value" % str(expr.impl)

        if self.unpack:
            rval = computed[self.outputs[0]]
        else:
            rval = [computed[o] for o in self.outputs]
        if self.reuse_computed:
            self.computed=computed
        return rval

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

    #PROTOCOLS:
    # 'python' - fn(*args) returns result (when n_inputs=1), returns results when n_inputs > 1
    #            This is the simplest protocol to use, and it is the default.
    # 'python_io' - fn(args, old_rval) returns same as 'python', but accepts previous return
    #               value from the same expr as second argument.  Permits reusing storage so
    #               it's typically faster than the 'python' protocol.
    # 'cond' - fn(args, old_rval, outputs).  Returns a dictionary of which inputs are required
    #          to compute which remaining outputs.  Stores computable outputs into the 'outputs'
    #          argument list prior to returning.  Permits implementing 'if' and 'switch' type
    #          expressions.
    fn_protocol = 'python'
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
        Update the meta-data attributes of input and output symbols.
        Return a set of symbols that were changed, or None if no changes were made.

        Raise TypeError() if inputs have become incompatible with expr.
        """
        pass

    def as_input(self, closure, obj, symbol_ctor=Symbol.new):
        """Return a Symbol in `closure` for `obj`.

        This is a helper method used by Impl.__call__.

        If `obj` is not a symbol, it will be wrapped in one.

        Raises ValueError on an obj of a foreign closure, does not catch any exceptions from
        type_cls.
        """
        if isinstance(obj, Symbol):
            if obj.closure is not closure:
                raise ValueError('Input in foreign closure', obj)
            return obj
        return symbol_ctor(closure=closure, value=obj)

    def outputs_from_inputs(self, inputs):
        """Return `self.n_output` Symbols for the outputs of an Expr of this Impl.

        This is a helper method used by Impl.__call__.
        Override this method to return different types of symbols with Impl.__call__.
        """
        return [Symbol.new(inputs[0].closure) for o in range(self.n_outputs)]

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
