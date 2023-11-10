import unittest

from bbo.utils.parameter_config import (
    ParameterType,
    ParameterConfig,
    SearchSpace,
)


class ParameterConfigTest(unittest.TestCase):
    def test_float_param(self):
        pc = ParameterConfig.factory(
            name='float',
            bounds=(0.0, 10.0),
        )

        self.assertEqual(pc.type, ParameterType.DOUBLE)
        self.assertEqual(pc.bounds, (0.0, 10.0))
        with self.assertRaises(ValueError):
            pc.feasible_values
        self.assertEqual(pc.num_feasible_values, float('inf'))

        v = pc.sample()
        self.assertTrue(pc.contains(v))
        self.assertFalse(pc.contains(11))

    def test_int_param(self):
        pc = ParameterConfig.factory(
            name='int',
            bounds=(0, 2),
        )

        self.assertEqual(pc.type, ParameterType.INTEGER)
        self.assertEqual(pc.bounds, (0, 2))
        self.assertEqual(pc.feasible_values, [0, 1, 2])
        self.assertEqual(pc.num_feasible_values, 3)

        v = pc.sample()
        self.assertTrue(pc.contains(v))
        self.assertFalse(pc.contains(3))

    def test_discrete_param(self):
        feasible_values = [0, 1, 3, 5, 6]
        pc = ParameterConfig.factory(
            name='discrete',
            feasible_values=feasible_values,
        )

        self.assertEqual(pc.type, ParameterType.DISCRETE)
        self.assertEqual(pc.bounds, (0, 6))
        self.assertEqual(pc.feasible_values, feasible_values)
        self.assertEqual(pc.num_feasible_values, len(feasible_values))

        v = pc.sample()
        self.assertTrue(pc.contains(v))
        self.assertFalse(pc.contains(-1))

    def test_categorical_param(self):
        feasible_values = ['a', 'b', 'c']
        pc = ParameterConfig.factory(
            name='categorical',
            feasible_values=feasible_values,
        )

        self.assertEqual(pc.type, ParameterType.CATEGORICAL)
        with self.assertRaises(ValueError):
            pc.bounds
        self.assertEqual(pc.feasible_values, feasible_values)
        self.assertEqual(pc.num_feasible_values, len(feasible_values))

        v = pc.sample()
        self.assertTrue(pc.contains(v))
        self.assertFalse(pc.contains('d'))

    def test_default_value(self):
        pc = ParameterConfig.factory(
            name='float',
            bounds=(0.0, 1.0),
            default_value=0.5,
        )
        self.assertEqual(pc.default_value, 0.5)


class SearchSpaceTest(unittest.TestCase):
    def test_params(self):
        sp = SearchSpace()
        sp.add_float_param('x0', 0, 1)
        sp.add_int_param('x1', 0, 1)
        sp.add_discrete_param('x2', [0, 2, 4])
        sp.add_categorical_param('x3', ['a', 'b', 'c'])

        for param_type in ParameterType:
            self.assertEqual(sp.num_parameters(param_type), 1)

    def test_duplicated_param(self):
        sp = SearchSpace()
        sp.add_int_param('x0', 0, 1)
        with self.assertRaises(ValueError):
            sp.add_int_param('x0', 0, 1)

    def test_discrete_param(self):
        sp = SearchSpace()
        with self.assertRaises(ValueError):
            sp.add_discrete_param('x0', ['a', 'b', 'c'])

    def test_categorical_param(self):
        sp = SearchSpace()
        with self.assertRaises(ValueError):
            sp.add_categorical_param('x0', [0, 1, 2])
