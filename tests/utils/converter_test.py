import unittest

import numpy as np
import torch

from bbo.utils.metric_config import MetricInformation, ObjectiveMetricGoal
from bbo.utils.trial import Trial
from bbo.utils.parameter_config import ScaleType, SearchSpace
from bbo.utils.converters.converter import (
    SpecType,
    DefaultInputConverter,
    DefaultOutputConverter,
    DefaultTrialConverter,
    GroupedFeatureTrialConverter,
    ArrayTrialConverter,
)
from bbo.utils.converters.torch_converter import (
    TorchGroupedFeatureTrialConverter,
    TorchArrayTrialConverter
)


def _create_search_space():
    sp = SearchSpace()
    sp.add_float_param('float', 0, 10)
    sp.add_float_param('float_linear', 0, 10, scale_type=ScaleType.LINEAR)
    sp.add_int_param('int', 1, 10)
    sp.add_discrete_param('discrete', [0, 2, 4, 6])
    sp.add_categorical_param('categorical', ['a', 'b', 'c'])
    return sp
    

class DefaultInputConverterTest(unittest.TestCase):
    def setUp(self):
        self.sp = _create_search_space()
        self.count = 3
        parameters = [self.sp.sample() for _ in range(self.count)]
        self.trials = [Trial(parameters=p) for p in parameters]

    def test_shape(self):
        converters = {
            name: DefaultInputConverter(pc)
            for name, pc in self.sp.parameter_configs.items()
        }
        for name, converter in converters.items():
            array = converter.convert(self.trials)
            self.assertEqual(array.shape, (self.count, 1))
            
            pv = converter.to_parameter_values(array)
            for i, v in enumerate(pv):
                self.assertEqual(self.trials[i].parameters[name], v)

    def test_float(self):
        pc = self.sp.get('float')
        converter = DefaultInputConverter(pc)
        output_spec = converter.output_spec
        self.assertEqual(output_spec.name, pc.name)
        self.assertEqual(output_spec.type, SpecType.DOUBLE)
        self.assertEqual(output_spec.bounds, pc.bounds)
        self.assertEqual(output_spec.scale_type, None)

    def test_float_linear(self):
        pc = self.sp.get('float_linear')
        converter = DefaultInputConverter(pc)
        output_spec = converter.output_spec
        self.assertEqual(output_spec.name, pc.name)
        self.assertEqual(output_spec.type, SpecType.DOUBLE)
        self.assertEqual(output_spec.bounds, (0, 1))
        self.assertEqual(output_spec.scale_type, None)

        array = converter.convert(self.trials)
        self.assertTrue(((array >= 0) & (array <= 1)).all())

    def test_int(self):
        pc = self.sp.get('int')
        converter = DefaultInputConverter(pc)
        output_spec = converter.output_spec
        self.assertEqual(output_spec.name, pc.name)
        self.assertEqual(output_spec.type, SpecType.INTEGER)
        self.assertEqual(output_spec.bounds, (0, len(pc.feasible_values)-1))
        self.assertEqual(output_spec.scale_type, None)

    def test_discrete(self):
        pc = self.sp.get('discrete')
        converter = DefaultInputConverter(pc)
        output_spec = converter.output_spec
        self.assertEqual(output_spec.name, pc.name)
        self.assertEqual(output_spec.type, SpecType.DISCRETE)
        self.assertEqual(output_spec.bounds, (0, len(pc.feasible_values)-1))
        self.assertEqual(output_spec.scale_type, None)

    def test_categorical(self):
        pc = self.sp.get('categorical')
        converter = DefaultInputConverter(pc)
        output_spec = converter.output_spec
        self.assertEqual(output_spec.name, pc.name)
        self.assertEqual(output_spec.type, SpecType.CATEGORICAL)
        self.assertEqual(output_spec.bounds, (0, len(pc.feasible_values)-1))
        self.assertEqual(output_spec.scale_type, None)


class DefaultOutputConverterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sp = _create_search_space()
        self.m = MetricInformation('obj', goal=ObjectiveMetricGoal.MAXIMIZE)
        self.count = 3
        parameters = [self.sp.sample() for _ in range(self.count)]
        self.trials = [Trial(parameters=p) for p in parameters]
        for i, t in enumerate(self.trials):
            t.complete({'obj': i})

    def test_creation(self):
        converter = DefaultOutputConverter(self.m)
        array = converter.convert(self.trials)
        self.assertTrue((array == np.array([0, 1, 2]).reshape(-1, 1)).all())


class TrialConverterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sp = _create_search_space()
        m = MetricInformation('obj', goal=ObjectiveMetricGoal.MAXIMIZE)
        self.input_converters = [DefaultInputConverter(pc) for pc in self.sp.parameters]
        self.output_converters = [DefaultOutputConverter(m), ]

        self.count = 3
        parameters = [self.sp.sample() for _ in range(self.count)]
        self.trials = [Trial(parameters=p) for p in parameters]
        for i, t in enumerate(self.trials):
            t.complete({'obj': i})

    def test_default_convert(self):
        convert_cls = DefaultTrialConverter
        converter = convert_cls(self.input_converters, self.output_converters)
        features, labels = converter.convert(self.trials)
        self.assertIsInstance(features, dict)
        self.assertIsInstance(labels, dict)
        trials = converter.to_trials(features, labels)
        self.assertEqual(trials, self.trials)

    def test_array_convert(self):
        convert_cls = ArrayTrialConverter
        converter = convert_cls(self.input_converters, self.output_converters)
        features, labels = converter.convert(self.trials)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        trials = converter.to_trials(features, labels)
        self.assertEqual(trials, self.trials)

        convert_cls = TorchArrayTrialConverter
        converter = convert_cls(self.input_converters, self.output_converters)
        features, labels = converter.convert(self.trials)
        self.assertIsInstance(features, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        trials = converter.to_trials(features, labels)
        self.assertEqual(trials, self.trials)

    def test_grouped_convert(self):
        for convert_cls in (GroupedFeatureTrialConverter, TorchGroupedFeatureTrialConverter):
            converter = convert_cls(self.input_converters, self.output_converters)
            features, labels = converter.convert(self.trials)
            for key in SpecType:
                self.assertIn(key.name, features)
            
            for v in features.values():
                if isinstance(converter, TorchGroupedFeatureTrialConverter):
                    self.assertIsInstance(v, torch.Tensor)
                elif isinstance(converter, GroupedFeatureTrialConverter):
                    self.assertIsInstance(v, np.ndarray)
                else:
                    raise ValueError
                
            trials = converter.to_trials(features, labels)
            self.assertEqual(trials, self.trials)
            