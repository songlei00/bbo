import unittest

from bbo.utils.metric_config import ObjectiveMetricGoal, MetricInformation


class MetricConfigTest(unittest.TestCase):
    def test_create(self):
        m = MetricInformation('obj', goal=ObjectiveMetricGoal.MAXIMIZE)
        self.assertEqual(m.name, 'obj')
        self.assertEqual(m.goal, ObjectiveMetricGoal.MAXIMIZE)

    def test_min_max(self):
        m = MetricInformation(
            'obj',
            goal=ObjectiveMetricGoal.MAXIMIZE,
            min_value=-5,
            max_value=5,
        )
        self.assertEqual(m.min_value, -5)
        self.assertEqual(m.max_value, 5)