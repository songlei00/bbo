from bbo.algorithms.designers import RandomDesigner
from bbo.benchmarks.experimenters import Branin2DExperimenter

exp = Branin2DExperimenter()
designer = RandomDesigner(exp.problem_statement())

for _ in range(10):
    suggestions = designer.suggest()
    exp.evaluate(suggestions)
    print(suggestions)