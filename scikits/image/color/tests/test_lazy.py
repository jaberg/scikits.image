#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for color conversion functions with lazy evaluation.


:license: modified BSD
"""

import os.path
import timeit

import numpy as np
from numpy.testing import *

from scikits.image.io import imread
from scikits.image.color import (
    rgb2hsv, hsv2rgb,
    rgb2xyz, xyz2rgb,
    rgb2rgbcie, rgbcie2rgb,
    convert_colorspace
    )

import scikits.image.color.lazyopts
scikits.image.color.lazyopts.disable()

from scikits.image import data_dir
from scikits import lazy

import colorsys

def assert_almost_equal(a,b,decimal=5):
    return np.testing.assert_almost_equal(a,b,decimal=decimal)


class TestColorconv(TestCase):

    img_rgb = imread(os.path.join(data_dir, 'color.png'))
    img_grayscale = imread(os.path.join(data_dir, 'camera.png'))

    colbars = np.array([[1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 0, 1, 0, 1, 0, 1, 0]])
    colbars_array = np.swapaxes(colbars.reshape(3, 4, 2), 0, 2)
    colbars_point75 = colbars * 0.75
    colbars_point75_array = np.swapaxes(colbars_point75.reshape(3, 4, 2), 0, 2)

    def test_lazy_convert_colorspace(self):
        colspaces = ['HSV', 'RGB CIE', 'XYZ']
        colfuncs_from = [hsv2rgb, rgbcie2rgb, xyz2rgb]
        colfuncs_to = [rgb2hsv, rgb2rgbcie, rgb2xyz]
        img = lazy.lnumpy.NdarraySymbol.new(value=self.colbars_array)

        assert_almost_equal(convert_colorspace(self.colbars_array, 'RGB',
                                               'RGB'), self.colbars_array)
        for i, space in enumerate(colspaces):
            gt = colfuncs_from[i](img)
            assert_almost_equal(convert_colorspace(self.colbars_array, space,
                                                  'RGB'), gt)

            gt = colfuncs_to[i](img)
            assert_almost_equal(convert_colorspace(self.colbars_array, 'RGB',
                                                   space), gt)

        self.assertRaises(ValueError, convert_colorspace, img, 'nokey', 'XYZ')
        self.assertRaises(ValueError, convert_colorspace, img, 'RGB', 'nokey')

    def test_opencl_implementation(self):
        scikits.image.color.lazyopts.enable()
        colspaces = ['HSV', 'RGB CIE', 'XYZ']
        colfuncs_from = [hsv2rgb, rgbcie2rgb, xyz2rgb]
        colfuncs_to = [rgb2hsv, rgb2rgbcie, rgb2xyz]

        # create an input, and make some promises about it.
        img = lazy.lnumpy.NdarraySymbol.new(value=self.colbars_array)
        img.value = img.value.astype('float32')
        img.contiguous=True
        img.shape = (None, None, 3)
        img.dtype=np.dtype('float32')

        assert_almost_equal(convert_colorspace(self.colbars_array, 'RGB',
                                               'RGB'), self.colbars_array)
        for i, space in enumerate(colspaces):
            print 'test: colorspace', space
            gt = colfuncs_from[i](img)
            eval_gt = lazy.function([img], gt)
            assert_almost_equal(convert_colorspace(self.colbars_array, space,
                                                  'RGB'), eval_gt(img.value))

            gt = colfuncs_to[i](img)
            eval_gt = lazy.function([img], gt)
            #eval_gt.print_eval_order()
            assert_almost_equal(convert_colorspace(self.colbars_array, 'RGB',
                                                   space), eval_gt(img.value))

    def test_opencl_speed(self):
        scikits.image.color.lazyopts.enable()
        colspaces = ['RGB CIE', 'XYZ']
        colfuncs_from = [ rgbcie2rgb, xyz2rgb]
        colfuncs_to = [ rgb2rgbcie, rgb2xyz]
        imgdata = np.random.RandomState(234).rand(1600,1200,3).astype('float32')

        # create an input, and make some promises about it.
        img = lazy.lnumpy.NdarraySymbol.new()
        img.contiguous=True
        img.shape = (None, None, 3)
        img.dtype=np.dtype('float32')

        for i, space in enumerate(colspaces):
            print 'test: colorspace', space
            gt = colfuncs_from[i](img)
            scikits.image.color.lazyopts.enable()
            fn_yes_cl = lazy.function([img], gt)
            scikits.image.color.lazyopts.disable()
            fn_no_cl = lazy.function([img], gt)

            fn_yes_cl.reuse_computed = True # optimization

            def with_cl():
                return fn_yes_cl(imgdata)
            def without_cl():
                return fn_no_cl(imgdata)
            def orig_fn():
                return convert_colorspace(imgdata, space, 'RGB')

            print 'with_cl times', timeit.Timer(with_cl).repeat(3,3)
            print 'without_cl times',timeit.Timer(without_cl).repeat(3,3)
            print 'orig_fn times', timeit.Timer(orig_fn).repeat(3,3)

if __name__ == "__main__":
    run_module_suite()

