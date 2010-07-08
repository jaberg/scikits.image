from scikits.lazy import register_transform, transform_db

from scikits.image.color import colorconv

_enabled = True
def enable():
    global _enabled
    _enabled=True
def disable():
    global _enabled
    _enabled=False

@register_transform(transform_db['begin_backend'].position + 0.1, 'default')
def replace_colorconvert_with_opencl(closure, **kwargs):
    if not _enabled: return
    for expr in closure.expr_iter_with_Impl(colorconv.ColorConvert3x3):
        print 'IMPL: ', expr
