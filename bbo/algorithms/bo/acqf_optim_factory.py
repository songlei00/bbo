def propose_rand_samples_sobol(dims, n, lb, ub):
    seed = np.random.randint(int(5e5))
    sobol = SobolEngine(dims, scramble=True, seed=seed) 
    cands = sobol.draw(n).to(dtype=torch.float64).cpu().detach().numpy()
    cands = cands * (ub - lb) + lb
    return cands


def optimize_acqf(dims, gpr, X_sample, Y_sample, n, lb, ub):
    # maximize acquisition function
    X = propose_rand_samples_sobol(dims, 1024, lb, ub)
    X_acqf = expected_improvement(gpr, X_sample, Y_sample, X, xi=0.0001, use_ei=True)
    # X_acqf = upper_confidence_bound(gpr, X_sample, Y_sample, X, beta=0.1)
    X_acqf = X_acqf.reshape(-1)
    indices = np.argsort(X_acqf)[-n: ]
    proposed_X, proposed_X_acqf = X[indices], X_acqf[indices]
    return proposed_X, proposed_X_acqf


@define
class AcqfOptimizer:
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement)
    )
    _optimizer: Designer = field
    _lr: float = field(default=0.01, validator=validators.instance_of(float), converter=float)
    _epochs: int = field(default=200, validator=validators.instance_of(int))
    _num_restarts: int = field(default=3, validator=validators.instance_of(int))
    _num_raw_samples: int = field(default=2048, validator=validators.instance_of(int))

    def optimize(self):
        dims = self._problem_statement.search_space.num_parameters()
        propose_rand_samples_sobol(dims, self._num_raw_samples, )
        for i in range(self._num_restarts):
            pass

@define
class AcqfOptimizerFactory:
    _acqf_optimizer_type: str = field(
        default='lbfgs', kw_only=True,
        validator=validators.in_(['random', 'adam', 'lbfgs', 'pso', 'nsgaii', 're'])
    )
    _optimizer_config: Dict = field(factory=dict)

    def __call__(self, converter: ):
        if self._acqf_optimizer_type == 'random':

