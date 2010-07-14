import unittest
from scikits.lazy import function, Symbol, Impl

@Impl.allow_lazy()
def dummy_fn(a):
    return a + 5

def func(*args, **kwargs):
    return function(
            transform_policy_ctor=lambda :(lambda x:x),
            *args,**kwargs) 

class Test_FunctionClosure_new_givens(unittest.TestCase):
    """Test the givens argument to FunctionClosure.new """

    # Behavioural
    # ===========
    # test that you can create an unattached symbol as a value, give it attributes
    # like a shape, and an allocation_target, and a name, and a 'strict=True' and those values
    # will be respected by the FunctionClosure
    def test_behaviour_0(self):
        s = Symbol.new(name='s')

        f = func([s], dummy_fn(s))
        assert f[s].name=='s'

        f = func([s], dummy_fn(s), givens={s:Symbol.new_kwargs(
            name='u', strict=True, shape=(4,5,6), opencl_ctx='blah')})
        assert f[s].name == 'u'
        assert f[s].strict==True
        assert f[s].shape == (4,5,6)
        assert f[s].opencl_ctx=='blah'


    # Unit tests
    # ==========

    # test that givens arg can be list, tuple or dict

    # test that it works recursively
    # .. test that you can replace a leaf or internal node
    # .. with a leaf or internal node

    # test that attributes are transferred properly
    # test that mutable and strict are transferred correctly

    # test that givens can replace inputs and outputs

    # test that it fails if you try to replace something twice

    # test that using a non-symbol key fails

    # test that using a non-symbol value fails w NotImplementedError

