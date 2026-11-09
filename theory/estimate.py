import math
import numpy as np
from scipy.optimize import least_squares, minimize, root

# Helper functions

def compute_group_sizes(alpha1, alpha0, beta):
    return [alpha1 * beta, (1 - alpha1) * beta, (1 - alpha0) * (1 - beta), alpha0 * (1 - beta)]

def random_unit_vector():
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)
    theta = np.arccos(costheta)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


class Estimate():
    def __init__(self, m1=2, m2=2, zero_tolerance=1e-8):
        self.m1 = m1
        self.m2 = m2
        self.zero_tolerance = zero_tolerance
        self.attributes = [
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1]
        ]

    def get_D(self, tau, m1, m2):
        return m1 + tau ** 2 * m2

    def get_Ns(self, tau, b, m1, m2):
        N1 = m1 + tau * m2 + b
        N2 = m1 - tau * m2 + b
        N3 = m1 - tau * m2 - b
        N4 = m1 + tau * m2 - b
        return [N1, N2, N3, N4]

    def get_f(self, va, tau, b):
        a1, a2 = va
        return (self.m1 + tau * a2 * self.m2 / a1 + a1 * b) / math.sqrt(2 * self.get_D(tau=tau, m1=self.m1, m2=self.m2))

    def get_gammas(self, params, tau, b, m1, m2):
        alpha1, alpha0, beta = params
        N1, N2, N3, N4 = self.get_Ns(tau=tau, b=b, m1=m1, m2=m2)
        D = self.get_D(tau=tau, m1=m1, m2=m2)
        gamma1 = alpha1 * beta * math.exp(-1 * N1 ** 2 / (2 * D))
        gamma2 = (1 - alpha1) * beta * math.exp(-1 * N2 ** 2 / (2 * D))
        gamma3 = (1 - alpha0) * (1 - beta) * math.exp(-1 * N3 ** 2 / (2 * D))
        gamma4 = alpha0 * (1 - beta) * math.exp(-1 * N4 ** 2 / (2 * D))
        return [gamma1, gamma2, gamma3, gamma4]

    def get_f_partials(self, va, tau, b):
        a1, a2 = va
        D = self.get_D(tau=tau, m1=self.m1, m2=self.m2)
        f_tau = (self.m2 * ((a2 - tau) * self.m1 - tau * a1 * b)) / (D * math.sqrt(2 * D))
        f_b = a1 / math.sqrt(2 * D)
        return f_tau, f_b

    def get_F_partials(self, params, tau, b):
        alpha1, alpha0, beta = params
        gamma1, gamma2, gamma3, gamma4 = self.get_gammas(params=params, tau=tau, b=b, m1=self.m1, m2=self.m2)
        N1, N2, N3, N4 = self.get_Ns(tau=tau, b=b, m1=self.m1, m2=self.m2)
        D = self.get_D(tau=tau, m1=self.m1, m2=self.m2)
        F1_tau = (tau - 1) * self.m2 / D ** 2 * (gamma1 * N1 * (tau * N1 - D) + gamma4 * N4 * (tau * N4 - D)) + (tau + 1) * self.m2 / D ** 2 * (gamma2 * N2 * (tau * N2 - D) + gamma3 * N3 * (tau * N3 - D)) + gamma1 + gamma2 + gamma3 + gamma4
        F2_tau = self.m2 / D ** 2 * (gamma1 * N1 * (tau * N1 - D) + gamma2 * N2 * (tau * N2 - D) - gamma3 * N3 * (tau * N3 - D) - gamma4 * N4 * (tau * N4 - D))
        F1_b = ((tau - 1) * (-1 * gamma1 * N1 + gamma4 * N4) + (tau + 1) * (-1 * gamma2 * N2 + gamma3 * N3)) / D
        F2_b = (-1 * gamma1 * N1 - gamma2 * N2 - gamma3 * N3 - gamma4 * N4) / D
        F1_alpha1 = (tau - 1) * beta * math.exp(-1 * N1 ** 2 / (2 * D)) - (tau + 1) * beta * math.exp(-1 * N2 ** 2 / (2 * D))
        F2_alpha1 = beta * math.exp(-1 * N1 ** 2 / (2 * D)) - beta * math.exp(-1 * N2 ** 2 / (2 * D))
        F1_alpha0 = (tau - 1) * (1 - beta) * math.exp(-1 * N4 ** 2 / (2 * D)) - (tau + 1) * (1 - beta) * math.exp(-1 * N3 ** 2 / (2 * D))
        F2_alpha0 = -1 * (1 - beta) * math.exp(-1 * N4 ** 2 / (2 * D)) + (1 - beta) * math.exp(-1 * N3 ** 2 / (2 * D))
        F1_beta = (tau - 1) * (alpha1 * math.exp(-1 * N1 ** 2 / (2 * D)) - alpha0 * math.exp(-1 * N4 ** 2 / (2 * D))) + (tau + 1) * ((1 - alpha1) * math.exp(-1 * N2 ** 2 / (2 * D)) - (1 - alpha0) * math.exp(-1 * N3 ** 2 / (2 * D)))
        F2_beta = alpha1 * math.exp(-1 * N1 ** 2 / (2 * D)) + (1 - alpha1) * math.exp(-1 * N2 ** 2 / (2 * D)) + alpha0 * math.exp(-1 * N4 ** 2 / (2 * D)) + (1 - alpha0) * math.exp(-1 * N3 ** 2 / (2 * D))
        return F1_tau, F2_tau, F1_b, F2_b, F1_alpha1, F2_alpha1, F1_alpha0, F2_alpha0, F1_beta, F2_beta

    def _compute_jacobian(self, alpha1, alpha0, beta):
        return np.array([
            [ beta,        0,       alpha1       ],
            [-beta,        0,       1 - alpha1   ],
            [   0,  -(1 - beta),  -(1 - alpha0) ],
            [   0,   1 - beta,    -alpha0       ]
        ])

    def solve_tau_b(self, params):

        def H(x):
            tau, b = x
            gamma1, gamma2, gamma3, gamma4 = self.get_gammas(params=params, tau=tau, b=b, m1=self.m1, m2=self.m2)
            eq1 = tau - (gamma1 + gamma4 - gamma2 - gamma3) / (gamma1 + gamma2 + gamma3 + gamma4)
            eq2 = gamma1 + gamma2 - gamma3 - gamma4
            return [eq1, eq2]

        initial_guesses = [
            [0.08, 0.],
            [0., 0.],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
        ]
        for guess in initial_guesses:
            result = root(H, guess)
            if result.success:
                break
        return result.x[0], result.x[1]


    def solve_derivative(self, params, tau, b):
        F1_tau, F2_tau, F1_b, F2_b, F1_alpha1, F2_alpha1, F1_alpha0, F2_alpha0, F1_beta, F2_beta = self.get_F_partials(params=params, tau=tau, b=b)
        def H0(x):
            tau_p, b_p = x
            eq1 = F1_tau * tau_p + F1_b * b_p + F1_alpha1
            eq2 = F2_tau * tau_p + F2_b * b_p + F2_alpha1
            val = [eq1, eq2]
            return val
        def H1(x):
            tau_p, b_p = x
            eq1 = F1_tau * tau_p + F1_b * b_p + F1_alpha0
            eq2 = F2_tau * tau_p + F2_b * b_p + F2_alpha0
            val = [eq1, eq2]
            return val
        def H2(x):
            tau_p, b_p = x
            eq1 = F1_tau * tau_p + F1_b * b_p + F1_beta
            eq2 = F2_tau * tau_p + F2_b * b_p + F2_beta
            val = [eq1, eq2]
            return val

        initial_guesses = [
            [0.08, 0.],
            [0., 0.],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
        ]
        all_tau_p, all_b_p = [], []
        for guess in initial_guesses:
            result_alpha1 = root(H0, guess)
            if result_alpha1.success:
                all_tau_p.append(result_alpha1.x[0])
                all_b_p.append(result_alpha1.x[1])

            result_alpha0 = root(H1, guess)
            if result_alpha0.success:
                all_tau_p.append(result_alpha0.x[0])
                all_b_p.append(result_alpha0.x[1])

            result_beta = root(H2, guess)
            if result_beta.success:
                all_tau_p.append(result_beta.x[0])
                all_b_p.append(result_beta.x[1])
        return all_tau_p, all_b_p

    def compute_acc(self, tau, b, m1, m2):
        acc = []
        N1, N2, N3, N4 = self.get_Ns(tau=tau, b=b, m1=m1, m2=m2)
        D = self.get_D(tau=tau, m1=m1, m2=m2)
        for N in [N1, N2, N3, N4]:
            acc.append(0.5 * (1 + math.erf(N / math.sqrt(2 * D))))
        return np.array(acc)

    def solve_ms(self, acc0, acc1, params0, params1):
        """Fit m1 and m2 from two adjusted accuracies and configurations."""
        def G(x):
            tau, b, tau_, b_, m1, m2 = x
            gamma1, gamma2, gamma3, gamma4 = self.get_gammas(params0, tau, b, m1, m2)
            eq1 = tau - (gamma1 + gamma4 - gamma2 - gamma3) / sum((gamma1, gamma2, gamma3, gamma4))
            eq2 = gamma1 + gamma2 - gamma3 - gamma4
            eq3 = acc0 - self.compute_acc(tau, b, m1, m2).mean()
            gamma1_, gamma2_, gamma3_, gamma4_ = self.get_gammas(params1, tau_, b_, m1, m2)
            eq4 = tau_ - (gamma1_ + gamma4_ - gamma2_ - gamma3_) / sum((gamma1_, gamma2_, gamma3_, gamma4_))
            eq5 = gamma1_ + gamma2_ - gamma3_ - gamma4_
            eq6 = acc1 - self.compute_acc(tau_, b_, m1, m2).mean()
            return eq1**2 + eq2**2 + 10 * eq3**2 + eq4**2 + eq5**2 + 10 * eq6**2

        # The reference constrains tau and b to be nonnegative; observed
        # allocations can require negative tau or b, so constrain only m.
        constraints = [{'type': 'ineq', 'fun': lambda x, i=i: x[i]} for i in (4, 5)]
        starts = ([0., 0., 0., 0., 2., 2.],
                  [0.1, 0., 0., 0., 2., 2.],
                  [0.1, 0., 0., 0., self.m1, self.m2])
        results = [minimize(G, start, constraints=constraints, method='SLSQP') for start in starts]
        result = min((r for r in results if r.success), key=lambda r: r.fun, default=results[0])
        if not result.success:
            raise RuntimeError(f"Separability fit failed: {result.message}")
        self.m1, self.m2 = map(float, result.x[4:6])
        return self.m1, self.m2, result

    def solve_ms_alter(self, grp_acc, params):
        """Fit m1 and m2 from one set of four group accuracies."""
        grp_acc = np.asarray(grp_acc, dtype=float)
        if grp_acc.shape != (4,):
            raise ValueError("Expected four group accuracies")

        def G(x):
            tau, b, m1, m2 = x
            gamma1, gamma2, gamma3, gamma4 = self.get_gammas(params, tau, b, m1, m2)
            eq1 = tau - (gamma1 + gamma4 - gamma2 - gamma3) / sum((gamma1, gamma2, gamma3, gamma4))
            eq2 = gamma1 + gamma2 - gamma3 - gamma4
            estimated_acc = self.compute_acc(tau, b, m1, m2)
            return (eq1 ** 2 + eq2 ** 2
                    + (estimated_acc[0] - grp_acc[0]) ** 2
                    + 10 * (estimated_acc[1] - grp_acc[1]) ** 2
                    + 10 * (estimated_acc[2] - grp_acc[2]) ** 2
                    + (estimated_acc[3] - grp_acc[3]) ** 2)

        constraints = [
            {'type': 'ineq', 'fun': lambda x, i=i: x[i]}
            for i in range(4)
        ]
        constraints.extend([
            {'type': 'ineq', 'fun': lambda x: 8 - x[2]},
            {'type': 'ineq', 'fun': lambda x: 8 - x[3]},
        ])
        result = minimize(
            G,
            [0., 0., 2., 2.],
            constraints=constraints,
            method='SLSQP',
        )
        if not result.success:
            raise RuntimeError(f"Separability fit failed: {result.message}")
        self.m1, self.m2 = map(float, result.x[2:4])
        return self.m1, self.m2, result

    def estimate_acc(self, params):
        tau, b = self.solve_tau_b(params=params)
        return self.compute_acc(tau=tau, b=b, m1=self.m1, m2=self.m2)

    def get_derivative(self, params):
        all_derivatives = np.zeros(3, dtype=float)
        tau, b = self.solve_tau_b(params=params)
        all_tau_p, all_b_p = self.solve_derivative(params=params, tau=tau, b=b)
        for va in self.attributes:
            a1, a2 = va
            f_tau, f_b = self.get_f_partials(va=va, tau=tau, b=b)
            f = self.get_f(va=va, tau=tau, b=b)
            A_f = math.exp(-1 * f ** 2) / math.sqrt(math.pi)
            for i in range(3):
                all_derivatives[i] += A_f * (f_tau * all_tau_p[i] + f_b * all_b_p[i])

        all_derivatives /= len(self.attributes)
        return all_derivatives

    def estimate_trajectory(self, params, delta_p=0.01, num_steps=50):
        params = np.array(params, dtype=float)

        param_traj = [params.copy()]
        acc_traj = [self.estimate_acc(params=params)]

        for _ in range(num_steps):
            all_derivatives = self.get_derivative(params=params)
            dev_norm = np.linalg.norm(all_derivatives)
            if dev_norm <= self.zero_tolerance:
                break
            params += delta_p * all_derivatives / dev_norm
            params = np.clip(params, 1e-6, 1 - 1e-6)

            param_traj.append(params.copy())
            acc_traj.append(self.estimate_acc(params=params))

        return np.array(acc_traj), np.array(param_traj)

    def baseline_trajectory(self, params, delta_p=0.01, traj_type='random', num_steps=50):
        """
        traj_type:
            "sign"   : sign of Grad
            "bal"    : direction toward fully balanced 4-group ratios
            "random" : random unit vector in parameter space
        """
        assert traj_type in ['sign', 'bal', 'random']

        params = np.array(params, dtype=float)

        param_traj = [params.copy()]
        acc_traj = [self.estimate_acc(params=params)]

        for _ in range(num_steps):
            all_derivatives = self.get_derivative(params=params)

            if traj_type == 'sign':
                direction = np.sign(all_derivatives)

            elif traj_type == 'bal':
                alpha1, alpha0, beta = params
                g0 = compute_group_sizes(alpha1, alpha0, beta)
                g_target = np.array([0.25, 0.25, 0.25, 0.25])
                v_g = g_target - g0

                J = self._compute_jacobian(alpha1, alpha0, beta)

                direction, _, _, _ = np.linalg.lstsq(J, v_g, rcond=None)

            elif traj_type == 'random':
                direction = random_unit_vector()

            norm = np.linalg.norm(direction)
            if norm <= self.zero_tolerance:
                break

            params += delta_p * direction / norm
            params = np.clip(params, 1e-6, 1 - 1e-6)

            param_traj.append(params.copy())
            acc_traj.append(self.estimate_acc(params=params))

        return np.array(acc_traj), np.array(param_traj)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Estimate subgroup accuracy trajectories")
    parser.add_argument("--m1", type=float, default=2.)
    parser.add_argument("--m2", type=float, default=2.)
    parser.add_argument("--params", type=float, nargs=3, default=[0.2, 0.3, 0.4], metavar=("ALPHA1", "ALPHA0", "BETA"))
    parser.add_argument("--delta-p", type=float, default=0.01)
    parser.add_argument("--num-steps", type=int, default=50)
    args = parser.parse_args()
    alpha1, alpha0, beta = args.params
    est = Estimate(m1=args.m1, m2=args.m2)
    estimated_adjusted_acc = est.estimate_acc([alpha1, alpha0, beta]).mean()
    acc_traj, param_traj = est.estimate_trajectory(params=args.params, delta_p=args.delta_p, num_steps=args.num_steps)
    print(f'Grad after updating {args.num_steps} steps. Acc: {acc_traj[-1].mean():.3f}, params: {param_traj[-1]}')

    acc_traj, param_traj = est.baseline_trajectory(params=args.params, delta_p=args.delta_p, num_steps=args.num_steps, traj_type='sign')
    print(f'Sign after updating {args.num_steps} steps. Acc: {acc_traj[-1].mean():.3f}, params: {param_traj[-1]}')
    acc_traj, param_traj = est.baseline_trajectory(params=args.params, delta_p=args.delta_p, num_steps=args.num_steps, traj_type='bal')
    print(f'Bal after updating {args.num_steps} steps. Acc: {acc_traj[-1].mean():.3f}, params: {param_traj[-1]}')
    acc_traj, param_traj = est.baseline_trajectory(params=args.params, delta_p=args.delta_p, num_steps=args.num_steps, traj_type='random')
    print(f'Random after updating {args.num_steps} steps. Acc: {acc_traj[-1].mean():.3f}, params: {param_traj[-1]}')
