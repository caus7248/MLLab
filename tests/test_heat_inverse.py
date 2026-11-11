import unittest
import numpy as np
from projects.heat_inverse.experiment import forward, fit_physics, dataset, run


class HeatInverseTests(unittest.TestCase):
    def test_initial_and_boundary_conditions(self):
        x = np.linspace(0, 1, 17)
        values = forward([0.1, 0.4], x, [0, 0.2])
        np.testing.assert_allclose(values[:, :, [0, -1]], 0, atol=1e-15)
        initial = np.sin(np.pi*x) + 0.3*np.sin(2*np.pi*x)
        np.testing.assert_allclose(values[:, 0], np.tile(initial, (2, 1)))

    def test_pde_residual(self):
        x = np.linspace(0.1, 0.9, 9)
        t, h, a = np.array([0.2, 0.4]), 1e-4, 0.23
        ut = (forward([a], x, t+h)-forward([a], x, t-h))/(2*h)
        uxx = (forward([a], x+h, t)-2*forward([a], x, t)+forward([a], x-h, t))/h**2
        np.testing.assert_allclose(ut, a*uxx, atol=1e-6)

    def test_noiseless_parameter_recovery(self):
        x, t = np.linspace(0, 1, 18)[1:-1], [0.1, 0.3, 0.6]
        truth = np.array([0.05, 0.2, 0.5, 0.8])
        data = forward(truth, x, t).reshape(4, -1)
        np.testing.assert_allclose(fit_physics(data, x, t), truth, atol=1e-7)

    def test_seed_reproduces_dataset(self):
        args = (10, (0.05, 0.5), [0.25, 0.75], [0.2], 0.01)
        first = dataset(np.random.default_rng(7), *args)
        second = dataset(np.random.default_rng(7), *args)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_reject_invalid_configuration(self):
        with self.assertRaises(ValueError):
            run(sensors=0)


if __name__ == "__main__":
    unittest.main()
