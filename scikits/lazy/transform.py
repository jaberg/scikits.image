"""
Registry and classes related to expression transformation
"""

class TransformHandle(object):
    def __init__(self, name, position, tags, transform_fn):
        self.name = name
        self.position=position
        self.tags=tags
        self.labels = [name] + list(tags)
        self.transform_fn = transform_fn
        self.enabled = True
    def __str__(self):
        return self.name
    def __repr__(self):
        return 'TransformHandle{%s}'%self.name
    def transform(self, expr_graph):
        if self.enabled:
            return self.transform_fn(expr_graph)

transform_db = {}

def register_transform(position, tags=[]):
    def deco(f):
        handle = TransformHandle(f.__name__,position,tags,f)
        if handle.name in transform_db:
            raise KeyError('name taken', handle.name)
        transform_db[handle.name] = handle
        return handle
    return deco

# DENOTE SOME KEY POINTS IN FULL OPTIMIZATION PIPELINE

transform_db['merge_0'] = TransformHandle('merge_0', 0, [], lambda x:x)

transform_db['begin_specialize'] = TransformHandle('begin_specialize', 10, [], lambda x:x)
transform_db['end_specialize'] = TransformHandle('end_specialize', 15, [], lambda x:x)

transform_db['begin_backend'] = TransformHandle('begin_backend', 20, [], lambda x:x)
transform_db['end_backend'] = TransformHandle('end_backend', 25, [], lambda x:x)

class TransformPolicy(object):
    @classmethod
    def new(cls, filter=lambda handle:True):
        handles = [(h.position,h) for name,h in transform_db.iteritems() if filter(h)]
        handles.sort()
        return cls([h[1] for h in handles])

    def info(self):
        print 'TransformPolicy<%i>'%id(self)
        for i,h in enumerate(self.handles):
            print ' ',i, h

    def __init__(self, handles):
        self.handles = handles

    def __call__(self, closure):
        for h in self.handles:
            h.transform(closure)


@register_transform(transform_db['merge_0'].position+0.01, 'default')
def merge_duplicate_constants(closure, **kwargs):
    #TODO:
    pass

@register_transform(transform_db['merge_0'].position+0.02, 'default')
def merge_duplicate_expressions(closure, **kwargs):
    #TODO:
    pass

@register_transform(1.0, 'default')
def infer_types(closure, **kwargs):
    """Do an initial pass of type inference
    """
    changed = set()
    for expr in closure.expr_iter():
        expr_changed = expr.impl.infer_type(expr)
        #TODO: propagate changes only
