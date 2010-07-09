
from . import lazy
from .lazy import (Impl, Symbol, Closure, IncrementalClosure, SpecializedClosure, Expr,
        function, UndefinedValue)

from .transform import register_transform, transform_db, TransformPolicy, TransformHandle

def symbol(value, name=None):
    return Symbol.new(lazy.default_closure, value=value, name=name)

def get_default_closure():
    return lazy.default_closure

def set_default_closure(closure):
    lazy.default_closure = closure

