"""Three-attribute Gaussian feature/configuration estimator.
"""

import itertools
import math
import numpy as np
from scipy.optimize import root, least_squares


def random_unit_vector_n3(dim):
    v = np.random.normal(size=dim)
    return v / (np.linalg.norm(v) + 1e-12)


def compute_group_sizes_n3(params):
    """
    params = [
        alpha2_pos, alpha2_neg,
        alpha3_pos, alpha3_neg,
        beta
    ]

    alpha2_pos = P(a2 = y | y = +1)
    alpha2_neg = P(a2 = y | y = -1)
    alpha3_pos = P(a3 = y | y = +1)
    alpha3_neg = P(a3 = y | y = -1)
    beta       = P(y = +1)

    Group order:
    (+,+,+), (+,+,-), (+,-,+), (+,-,-),
    (-,+,+), (-,+,-), (-,-,+), (-,-,-)
    """
    alpha2_pos, alpha2_neg, alpha3_pos, alpha3_neg, beta = params

    return np.array([
        beta * alpha2_pos * alpha3_pos,
        beta * alpha2_pos * (1 - alpha3_pos),
        beta * (1 - alpha2_pos) * alpha3_pos,
        beta * (1 - alpha2_pos) * (1 - alpha3_pos),

        (1 - beta) * (1 - alpha2_neg) * (1 - alpha3_neg),
        (1 - beta) * (1 - alpha2_neg) * alpha3_neg,
        (1 - beta) * alpha2_neg * (1 - alpha3_neg),
        (1 - beta) * alpha2_neg * alpha3_neg,
    ], dtype=float)

class EstimateN3:
    def __init__(self, m1=2.0, m2=2.0, m3=2.0, zero_tolerance=1e-8):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.ms = np.array([m1, m2, m3], dtype=float)
        self.zero_tolerance = zero_tolerance

        self.attributes = np.array(
            list(itertools.product([1, -1], repeat=3)),
            dtype=float
        )


    def get_D(self, tau1, tau2, ms):
        m1, m2, m3 = ms
        return m1 + tau1 ** 2 * m2 + tau2 ** 2 * m3

    def get_Ns(self, a, tau1, tau2, b, ms):
        """
        Correct-class margin for N=3.

        N_a = y(w^T z + b)
            = m1 + tau1 * (a1*a2) * m2
                + tau2 * (a1*a3) * m3
                + a1 * b
        """
        m1, m2, m3 = ms
        a1, a2, a3 = a

        return (
            m1
            + tau1 * (a1 * a2) * m2
            + tau2 * (a1 * a3) * m3
            + a1 * b
        )

    def get_gammas(self, params, tau1, tau2, b, ms):
        """
        gamma_a = r_a exp(-N_a^2 / (2D))
        """
        ratios = compute_group_sizes_n3(params)
        D = self.get_D(tau1, tau2, ms)

        gammas = []
        for r, a in zip(ratios, self.attributes):
            N = self.get_Ns(a, tau1, tau2, b, ms)
            gammas.append(r * math.exp(-(N ** 2) / (2 * D)))

        return np.array(gammas, dtype=float)

    def H_system(self, x, params, ms):
        """
        Theorem 2 equation system.

        x = [tau1, tau2, b]

        tau1 corresponds to feature z2.
        tau2 corresponds to feature z3.
        """
        tau1, tau2, b = x
        gammas = self.get_gammas(params, tau1, tau2, b, ms)

        denom = gammas.sum() + 1e-12

        a1 = self.attributes[:, 0]
        a2 = self.attributes[:, 1]
        a3 = self.attributes[:, 2]

        eq1 = tau1 - np.sum(a1 * a2 * gammas) / denom
        eq2 = tau2 - np.sum(a1 * a3 * gammas) / denom
        eq3 = np.sum(a1 * gammas)

        return np.array([eq1, eq2, eq3], dtype=float)

    def solve_taus_b(self, params):
        guesses = [
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0],
            [0.2, 0.2, 0.0],
            [-0.1, 0.1, 0.0],
            [0.1, -0.1, 0.0],
            [0.2, 0.2, 0.2],
            [-0.2, -0.2, -0.2],
        ]

        best = None
        best_norm = float("inf")

        for guess in guesses:
            result = root(lambda x: self.H_system(x, params, self.ms), guess)
            norm = np.linalg.norm(self.H_system(result.x, params, self.ms))

            if result.success and norm < best_norm:
                best = result.x
                best_norm = norm

        if best is None:
            # Fallback: least-squares solve if root fails.
            result = least_squares(
                lambda x: self.H_system(x, params, self.ms),
                x0=np.array([0.0, 0.0, 0.0]),
                bounds=([-5.0, -5.0, -20.0], [5.0, 5.0, 20.0])
            )
            best = result.x

        return best[0], best[1], best[2]

    def compute_acc(self, tau1, tau2, b, ms):
        D = self.get_D(tau1, tau2, ms)
        acc = []

        for a in self.attributes:
            N = self.get_Ns(a, tau1, tau2, b, ms)
            f = N / math.sqrt(2 * D)
            acc.append(0.5 * (1.0 + math.erf(f)))

        return np.array(acc, dtype=float)

    def estimate_acc(self, params):
        tau1, tau2, b = self.solve_taus_b(params)
        return self.compute_acc(tau1, tau2, b, self.ms)

    def solve_ms(self, accs, params_list):
        """Estimate m1, m2, and m3 from three performance observations."""
        accs = np.asarray(accs, dtype=float)
        params_list = [np.array(p, dtype=float) for p in params_list]
        if accs.shape != (3,) or len(params_list) != 3:
            raise ValueError("Expected exactly three accuracies and three configurations")
        if any(params.shape != (5,) for params in params_list):
            raise ValueError("Each N3 configuration must contain five parameters")

        def unpack(x):
            ms = x[:3]
            clf_params = []
            cursor = 3
            for _ in range(3):
                tau1 = x[cursor]
                tau2 = x[cursor + 1]
                b = x[cursor + 2]
                cursor += 3
                clf_params.append((tau1, tau2, b))

            return ms, clf_params

        def residuals(x):
            ms, clf_params = unpack(x)
            res = []

            for acc_obs, params, clf in zip(accs, params_list, clf_params):
                tau1, tau2, b = clf

                # Theorem-2 classifier equations.
                res.extend(self.H_system([tau1, tau2, b], params, ms))

                # Match observed adjusted accuracy.
                acc_pred = self.compute_acc(tau1, tau2, b, ms).mean()
                res.append(math.sqrt(10) * (acc_obs - acc_pred))

            return np.array(res, dtype=float)

        x0 = list(self.ms) + [0.0, 0.0, 0.0] * 3
        lower = [1e-5] * 3 + [-5.0, -5.0, -20.0] * 3
        upper = [50.0] * 3 + [5.0, 5.0, 20.0] * 3

        result = least_squares(
            residuals,
            x0=np.array(x0, dtype=float),
            bounds=(np.array(lower), np.array(upper)),
            loss="soft_l1",
            max_nfev=5000,
        )

        ms, clf_params = unpack(result.x)

        self.m1, self.m2, self.m3 = ms
        self.ms = ms

        return ms[0], ms[1], ms[2], clf_params, result

    def _jacobian(self, func, x, eps=1e-5):
        x = np.array(x, dtype=float)
        y0 = np.array(func(x), dtype=float)
        J = np.zeros((len(y0), len(x)), dtype=float)

        for i in range(len(x)):
            step = eps * (1.0 + abs(x[i]))
            xp = x.copy()
            xm = x.copy()
            xp[i] += step
            xm[i] -= step
            J[:, i] = (np.array(func(xp)) - np.array(func(xm))) / (2 * step)

        return J

    def _grad_scalar(self, func, x, eps=1e-5):
        x = np.array(x, dtype=float)
        grad = np.zeros(len(x), dtype=float)

        for i in range(len(x)):
            step = eps * (1.0 + abs(x[i]))
            xp = x.copy()
            xm = x.copy()
            xp[i] += step
            xm[i] -= step
            grad[i] = (func(xp) - func(xm)) / (2 * step)

        return grad

    def get_derivative(self, params):
        """
        Returns d adjusted-accuracy / d params.

        params are:
        [alpha2_pos, alpha2_neg, alpha3_pos, alpha3_neg, beta]
        """
        params = np.array(params, dtype=float)

        tau1, tau2, b = self.solve_taus_b(params)
        x0 = np.array([tau1, tau2, b], dtype=float)

        # H(x, theta) = 0
        H_x = self._jacobian(
            lambda x: self.H_system(x, params, self.ms),
            x0
        )

        H_theta = self._jacobian(
            lambda theta: self.H_system(x0, theta, self.ms),
            params
        )

        # dx/dtheta = - H_x^{-1} H_theta
        try:
            dx_dtheta = -np.linalg.solve(H_x, H_theta)
        except np.linalg.LinAlgError:
            dx_dtheta = -np.linalg.lstsq(H_x, H_theta, rcond=None)[0]

        def adjusted_acc_from_x(x):
            return self.compute_acc(x[0], x[1], x[2], self.ms).mean()

        dA_dx = self._grad_scalar(adjusted_acc_from_x, x0)

        # dA/dtheta = dA/dx dx/dtheta
        dA_dtheta = dA_dx @ dx_dtheta

        return dA_dtheta

    def _compute_jacobian(self, params, eps=1e-5):
        params = np.array(params, dtype=float)
        J = np.zeros((8, 5), dtype=float)

        for i in range(5):
            step = eps * (1.0 + abs(params[i]))
            pp = params.copy()
            pm = params.copy()
            pp[i] += step
            pm[i] -= step

            pp = np.clip(pp, 1e-8, 1 - 1e-8)
            pm = np.clip(pm, 1e-8, 1 - 1e-8)

            J[:, i] = (compute_group_sizes_n3(pp) - compute_group_sizes_n3(pm)) / (2 * step)

        return J

    def estimate_trajectory(self, params, delta_p=0.01, num_steps=50):
        params = np.array(params, dtype=float)

        ratio_traj = [compute_group_sizes_n3(params)]
        param_traj = [params.copy()]
        acc_traj = [self.estimate_acc(params=params)]

        for _ in range(num_steps):
            all_derivatives = self.get_derivative(params)
            dev_norm = np.linalg.norm(all_derivatives)
            if dev_norm <= self.zero_tolerance:
                break
            params += delta_p * all_derivatives / dev_norm
            params = np.clip(params, 1e-6, 1 - 1e-6)

            ratio_traj.append(compute_group_sizes_n3(params))
            param_traj.append(params.copy())
            acc_traj.append(self.estimate_acc(params=params))

        return np.array(acc_traj), np.array(param_traj)

    def baseline_trajectory(self, params, delta_p=0.01, traj_type="random", num_steps=50):
        """
        traj_type:
            "sign"   : sign of Grad
            "bal"    : direction toward fully balanced 8-group ratios
            "random" : random unit vector in parameter space
        """
        assert traj_type in ["sign", "bal", "random"]

        params = np.array(params, dtype=float)

        ratio_traj = [compute_group_sizes_n3(params=params)]
        param_traj = [params.copy()]
        acc_traj = [self.estimate_acc(params=params)]

        for _ in range(num_steps):
            all_derivatives = self.get_derivative(params=params)

            if traj_type == "sign":
                direction = np.sign(all_derivatives)

            elif traj_type == "bal":
                g_target = np.ones(8) / 8.0
                v_g = g_target - compute_group_sizes_n3(params=params)

                J = self._compute_jacobian(params=params)

                direction, _, _, _ = np.linalg.lstsq(J, v_g, rcond=None)

            elif traj_type == "random":
                direction = random_unit_vector_n3(5)

            norm = np.linalg.norm(direction)
            if norm <= self.zero_tolerance:
                break

            params += delta_p * direction / norm
            params = np.clip(params, 1e-6, 1 - 1e-6)

            ratio_traj.append(compute_group_sizes_n3(params=params))
            param_traj.append(params.copy())
            acc_traj.append(self.estimate_acc(params=params))

        return np.array(acc_traj), np.array(param_traj)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Three-attribute Gaussian estimate")
    parser.add_argument("--m1", type=float, default=2.)
    parser.add_argument("--m2", type=float, default=2.)
    parser.add_argument("--m3", type=float, default=2.)
    parser.add_argument("--params", type=float, nargs=5, default=[0.976, 0.184, 0.976, 0.184, 0.128])
    parser.add_argument("--delta-p", type=float, default=0.01)
    parser.add_argument("--num-steps", type=int, default=0)
    args = parser.parse_args()
    estimator = EstimateN3(m1=args.m1, m2=args.m2, m3=args.m3)
    params = args.params
    accuracies = estimator.estimate_acc(params)
    print("Estimated group accuracies:", accuracies)
    print("Adjusted accuracy:", accuracies.mean())
    if args.num_steps:
        trajectory, allocations = estimator.estimate_trajectory(params, args.delta_p, args.num_steps)
        print("Final adjusted accuracy:", trajectory[-1].mean())
        print("Final allocation:", allocations[-1])


