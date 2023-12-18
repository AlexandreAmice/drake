import numpy as np
import scipy.sparse as sp
import unittest

import pydrake.planning as mut
from pydrake.common.test_utilities import numpy_compare
from pydrake.geometry.optimization import HPolyhedron
from pydrake.common import RandomGenerator

class TestIrisFromCliqueCover(unittest.TestCase):

    def test_point_sampler_base_subclassable(self):
        class UniformHPolyhedronSampler(mut.PointSamplerBase):
            def __init__(self, set, generator):
                mut.PointSamplerBase.__init__(self)
                self.set = set
                self.generator = generator

            def DoSamplePoints(self, num_points):
                ret = np.zeros((self.set.ambient_dimension(), num_points))
                for i in range(num_points):
                    ret[:, i] = self.set.UniformSample(self.generator)
                return ret

        A = np.array([[-1,0], [0,-1], [1,1]])
        b = np.array([0,0,1])
        set = HPolyhedron(A,b)
        generator = RandomGenerator(0)
        sampler = UniformHPolyhedronSampler(set, generator)
        numpy_compare.assert_equal(sampler.set.A(), A)
        numpy_compare.assert_equal(sampler.set.b(), b)

        num_points = 3
        self.assertEqual(sampler.SamplePoints(num_points).shape, (2,3))
