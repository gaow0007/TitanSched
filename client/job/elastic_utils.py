
import numpy as np


class SpeedupFunction(object):

    def __init__(self, goodput_fn, max_batch_size=None, atomic_bsz_range=None,
                 accumulation=False, mem_size=32):
        self._goodput_fn = goodput_fn
        self._max_batch_size = max_batch_size
        self._atomic_bsz_range = atomic_bsz_range
        self._accumulation = accumulation
        self._mem_size = mem_size
        self._base_goodput, _, _ = goodput_fn.optimize(
            num_nodes=1, num_replicas=1, max_batch_size=max_batch_size,
            atomic_bsz_range=atomic_bsz_range, accumulation=accumulation)
        # Memoization for fast repeated queries.
        self._mem_speedup = -np.ones((mem_size, mem_size))
        self._mem_speedup[0, 0] = 0.0

    def __call__(self, num_nodes, num_replicas):
        assert np.all(np.less_equal(0, num_nodes))
        assert np.all(np.less_equal(num_nodes, num_replicas))
        assert np.all((num_nodes > 0) == (num_replicas > 0))
        # Remember what the output shape/format should be and flatten inputs.
        output_scalar = np.isscalar(num_nodes) and np.isscalar(num_replicas)
        output_shape = np.broadcast(num_nodes, num_replicas).shape
        num_nodes = np.broadcast_to(num_nodes, output_shape).flatten()
        num_replicas = np.broadcast_to(num_replicas, output_shape).flatten()
        # Return values which will be filled out.
        speedup = -np.ones(output_shape).flatten()
        # Fill in any previously memoized results first.
        indices = num_replicas < self._mem_size
        mem_idx = (num_nodes[indices], num_replicas[indices])
        speedup[indices] = self._mem_speedup[mem_idx]
        # Find the missing indices which still need to be computed.
        missing = speedup < 0
        if np.count_nonzero(missing) > 0:
            num_nodes, num_replicas = num_nodes[missing], num_replicas[missing]
            # Find unique inputs to reduce compuation.
            (num_nodes, num_replicas), inverse = np.unique(
                    np.stack([num_nodes, num_replicas]),
                    axis=1, return_inverse=True)
            goodput, _, _ = self._goodput_fn.optimize(
                num_nodes, num_replicas,
                max_batch_size=self._max_batch_size,
                atomic_bsz_range=self._atomic_bsz_range,
                accumulation=self._accumulation)
            # Memoize results.
            indices = num_replicas < self._mem_size
            mem_idx = (num_nodes[indices], num_replicas[indices])
            self._mem_speedup[mem_idx] = goodput[indices] / self._base_goodput
            # Fill in computed results.
            speedup[missing] = goodput[inverse] / self._base_goodput
        assert np.all(np.less_equal(0, speedup))
        speedup = speedup.reshape(output_shape)
        return speedup.item() if output_scalar else speedup




import autograd
import numpy as np
import collections
import scipy.optimize
import scipy.stats


PerfParams = collections.namedtuple("PerfParams", [
    # T_compute ~ alpha_c + beta_c * local_bsz +
    #             (alpha_a + beta_a * local_bsz) * accumulation_steps
    "alpha_c",  # Constant term of compute time
    "beta_c",   # Multiplicative factor of compute time
    # If inter-node: T_network ~ alpha_n + beta_n * replicas
    "alpha_n",  # Constant term of inter-node network time
    "beta_n",   # Retrogression factor of inter-node network time
    # If intra-node: T_network ~ alpha_r + beta_r * replicas
    "alpha_r",  # Constant term of intra-node network time
    "beta_r",   # Retrogression factor of intra-node network time
    # T_step ~ (T_compute ^ gamma + T_network ^ gamma) ^ (1 / gamma)
    # Essentially is a p-norm where p = gamma. When p ~ 1 then
    # T_step ~ T_compute + T_network, indicating no overlap between compute
    # and network. When p -> infinity then T_step = max(T_compute, T_network),
    # indicating perfect overlap. We limit gamma to [1, 10] since 10 is close
    # enough to approximate the max function for our purposes.
    "gamma",    # Models the degree of overlap between compute and network
])

GradParams = collections.namedtuple("GradParams", ["sqr", "var"])


class GoodputFunction(object):

    def __init__(self, perf_params, grad_params, init_batch_size):
        self._perf_params = PerfParams(*perf_params)
        self._grad_params = GradParams(*grad_params)
        self._init_batch_size = init_batch_size

    def __call__(self, num_nodes, num_replicas, atomic_bsz, accum_steps):
        return self.evaluate(num_nodes, num_replicas, atomic_bsz, accum_steps)

    def evaluate(self, num_nodes, num_replicas, atomic_bsz, accum_steps):
        batch_size = num_replicas * atomic_bsz * (accum_steps + 1)
        assert np.all(self._init_batch_size <= batch_size)
        return self.throughput(num_nodes, num_replicas, atomic_bsz,
                               accum_steps) * self.efficiency(batch_size)

    def throughput(self, num_nodes, num_replicas, atomic_bsz, accum_steps):
        accum_time = _predict_accum_time(self._perf_params, atomic_bsz)
        if np.isscalar(accum_time): 
            print('accum_time {}'.format(accum_time))
        network_time = _predict_network_time(self._perf_params,
                                             num_nodes, num_replicas)
        if np.isscalar(network_time): 
            print('network_time {}'.format(network_time))
        optim_time = np.exp(_predict_log_optim_time(self._perf_params,
                                                    accum_time, network_time))
        total_time = accum_steps * accum_time + optim_time
        batch_size = num_replicas * atomic_bsz * (accum_steps + 1)
        return batch_size / total_time

    def efficiency(self, batch_size):
        # Gradient noise scale (PGNS) is grad_var / grad_sqr * init_batch_size.
        # The following equals (PGNS + init_batch_size) / (PGNS + batch_size).
        grad_sqr = self._grad_params.sqr
        grad_var = self._grad_params.var
        scale = batch_size / self._init_batch_size
        denom = grad_var / scale + grad_sqr
        gain = np.where(denom > 0, (grad_var + grad_sqr) / denom, 1.0)
        return gain / scale

    def optimize(self, num_nodes, num_replicas, max_batch_size=None,
                 atomic_bsz_range=None, accumulation=False):
        assert np.all(np.less_equal(1, num_nodes))
        assert np.all(np.less_equal(num_nodes, num_replicas))
        if max_batch_size is None:
            max_batch_size = self._init_batch_size
        assert self._init_batch_size <= max_batch_size
        atomic_bsz_range = atomic_bsz_range or (None, None)
        min_atomic_bsz = atomic_bsz_range[0] or 1
        max_atomic_bsz = atomic_bsz_range[1] or max_batch_size
        # Remember what the output shape/format should be and flatten inputs.
        output_shape = np.broadcast(num_nodes, num_replicas).shape
        output_scalar = np.isscalar(num_nodes) or np.isscalar(num_replicas)
        num_nodes = np.broadcast_to(num_nodes, output_shape).flatten()
        num_replicas = np.broadcast_to(num_replicas, output_shape).flatten()
        # Samples 50 different total batch sizes in geometric space.
        min_batch_size = np.maximum(self._init_batch_size,
                                    min_atomic_bsz * num_replicas)
        batch_size = np.geomspace(min_batch_size, max_batch_size)
        local_bsz = batch_size / num_replicas
        eps = 1e-8  # Tolerance for floor/ceil operations.
        if accumulation:
            # If local_bsz size exceeds the max atomic batch size, split it
            # into a number of batches to form (atomic_bsz, accum_steps) such
            # that (atomic_bsz * (accum_steps + 1)) is close to local_bsz.
            #
            # If num_replicas == 1 and local_bsz > self._init_batch_size, then
            # set accum_steps to at least 1. This is because the gradient
            # statistics used for scaling up the learning rate are inaccurate
            # when there is only one atomic minibatch to estimate them from.
            accum_steps = np.ceil(local_bsz / max_atomic_bsz - eps) - 1
            accum_steps = np.where(
                np.logical_and(num_replicas == 1,
                               local_bsz > self._init_batch_size + eps),
                np.maximum(accum_steps, 1), accum_steps).astype(int)
        else:
            accum_steps = np.zeros_like(local_bsz, dtype=np.int)
        atomic_bsz = np.ceil(local_bsz / (accum_steps + 1) - eps).astype(int)
        # Evaluate the goodput of all candidate configurations.
        goodput = self.evaluate(num_nodes, num_replicas,
                                atomic_bsz, accum_steps)
        # Set the goodput of invalid configurations to 0.0.
        goodput = np.where((min_atomic_bsz <= atomic_bsz) &
                           (atomic_bsz <= max_atomic_bsz), goodput, 0.0)
        # Find the indices of the best configurations.
        indices = np.argmax(goodput, axis=0), np.arange(goodput.shape[1])
        # Restore the correct output shape and return results.
        goodput = goodput[indices].reshape(output_shape)
        atomic_bsz = atomic_bsz[indices].reshape(output_shape)
        accum_steps = accum_steps[indices].reshape(output_shape)
        if output_scalar:
            goodput = goodput.item()
            atomic_bsz = atomic_bsz.item()
            accum_steps = accum_steps.item()
        return goodput, atomic_bsz, accum_steps


def fit_perf_params(num_nodes, num_replicas, atomic_bsz,
                    accum_step_time, optim_step_time):
    # Fit the performance model given accum time and optim time measurements
    # for different configurations of num_nodes, num_replicas, and atomic_bsz.

    # HACK: We want to use the original numpy module for calls from the
    # SpeedupFunction for performance reasons, but also need those functions to
    # use autograd.numpy when we want to differentiate them. We patch the
    # global np reference only for the code invoked rom this function.
    global np  # Replace numpy from autograd.
    orig_np = np
    np = autograd.numpy

    num_nodes = np.array(num_nodes)
    num_replicas = np.array(num_replicas)
    local_bsz = np.array(atomic_bsz)
    accum_step_time = np.array(accum_step_time)
    optim_step_time = np.array(optim_step_time)

    # Set initial params to reasonable values.
    params = [1e-1, 1e-2] * 3 + [1.0 + 1e-3]
    # Set lower/upper bounds for each parameter. Add a small slack to lower
    # bounds to avoid numerical instability issues.
    lower = [1e-8, 1e-8] * 3 + [1.0]
    upper = [np.inf, np.inf] * 3 + [10.0]
    if len(np.unique(atomic_bsz)) == 1:
        # Fix alpha_c if only observed a single atomic batch size.
        # This makes the speedup model optimistic with respect to
        # scaling up the batchsize. This will assign equal weight
        # to the constant and multplicative factors for accum time
        # if there is only a single datapoint (which is by far the
        # most likely case for this scenario)
        params[0] = upper[0] = lower[0] = np.mean(accum_step_time) / 2
    if not np.any(num_nodes > 1):
        # Fix alpha_n and beta_n if no multi-node observations.
        params[2] = upper[2] = lower[2]
        params[3] = upper[3] = lower[3]
    if not np.any(np.logical_and(num_nodes == 1, num_replicas > 1)):
        # Fix alpha_r and beta_r if no single-node/multi-replica observations.
        params[4] = upper[4] = lower[4]
        params[5] = upper[5] = lower[5]
    if not np.any(num_replicas > 2):
        # Fix beta_n and beta_r if no replicas > 2.
        params[3] = upper[3] = lower[3]
        params[5] = upper[5] = lower[5]
    bounds = scipy.optimize.Bounds(lower, upper, keep_feasible=True)
    args = (num_nodes, num_replicas, atomic_bsz,
            accum_step_time, optim_step_time)
    # FIXME: need to handle optimization failures and propagate to the Trainer.
    grad_fn = autograd.grad(_obj_fn)
    result = scipy.optimize.minimize(_obj_fn, params, args=args,
                                     jac=grad_fn, bounds=bounds)
    params = result.x
    if not any(num_nodes > 1):
        # Enforce prior: alpha_n and beta_n are at least alpha_r and beta_r.
        params[2] = max(params[2], params[4] * 1.1)
        params[3] = max(params[3], params[5] * 1.1)
    np = orig_np  # Restore original numpy.
    return PerfParams(*params)


def _rmse(pred, true):
    return np.sqrt(((pred - true) ** 2).mean())


def _obj_fn(params, num_nodes, num_replicas, atomic_bsz,
            accum_step_time, optim_step_time):
    params = PerfParams(*params)
    pred_accum = _predict_accum_time(params, atomic_bsz)
    pred_network = _predict_network_time(params, num_nodes, num_replicas)
    pred_log_optim = _predict_log_optim_time(params, pred_accum, pred_network)
    # RMSLError of accum step time predictions.
    err1 = _rmse(np.log(pred_accum), np.log(accum_step_time))
    # RMSLError of optim step time predictions.
    err2 = _rmse(pred_log_optim, np.log(optim_step_time))
    # L2 regularization towards a smaller gamma, because it's easier to
    # optimize the alpha and beta parameters when gamma is smaller.
    reg1 = 1e-3 * (params.gamma - 1) ** 2
    # Penalize retrogression terms to prefer a more optimistic model.
    reg2 = 1e-2 * ((params.beta_n / params.alpha_n) ** 2 +
                   (params.beta_r / params.alpha_r) ** 2)
    return err1 + err2 + reg1 + reg2


def _predict_accum_time(params, atomic_bsz):
    params = PerfParams(*params)
    # Forward/backward passes should scale linearly with the batch size.
    return params.alpha_c + params.beta_c * atomic_bsz


def _predict_log_optim_time(params, accum_time, network_time):
    gamma = PerfParams(*params).gamma
    return np.log(accum_time ** gamma + network_time ** gamma) / gamma


def _predict_network_time(params, num_nodes, num_replicas):
    params = PerfParams(*params)
    # Select the most significant link between replicas, currently either
    # inter-node (nodes > 1) or intra-node (replicas > 1). Note that if
    # replicas == 1 then neither of these two conditions are matched.
    conds = [num_nodes > 1, num_replicas > 1]
    # Bandwidth is bottlenecked by the most significant link, alpha models
    # the overhead of transferring data across that link.
    bottleneck = np.select(conds, [params.alpha_n, params.alpha_r], 1e-8)
    # Assuming ring all-reduce, communication happens in a number of rounds
    # equal to the number of replicas. beta models the performance
    # retrogression from increasing the number of replicas beyond 2.
    retrogress = np.select(conds, [params.beta_n, params.beta_r], 1e-8)
    retrogress = retrogress * np.maximum(num_replicas - 2, 1e-8)
    return (bottleneck + retrogress)

