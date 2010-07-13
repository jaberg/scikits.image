import unittest
from scikits.lazy import FunctionClosure

class Test_FunctionClosure_new_givens(unittest.TestCase):
    """Test the givens argument to FunctionClosure.new """

    # Behavioural
    # ===========
    # test that you can create an unattached symbol as a value, give it attributes
    # like a shape, and an allocation_target, and a name, and a 'strict=True' and those values
    # will be respected by the FunctionClosure


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

