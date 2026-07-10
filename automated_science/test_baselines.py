import numpy as np

from run_baselines import integrate_rhs, make_predictor_rhs, safe_mse, split_train_test


class LinearDecayPredictor:
    def __init__(self, k):
        self.k = k

    def predict(self, states):
        states = np.asarray(states)
        return -self.k * states[:, 0]


def main():
    t = np.linspace(0.0, 1.0, 16)
    x = np.exp(-0.4 * t)[:, None]
    t_train, x_train, t_test, x_test, split_idx = split_train_test(t, x)
    assert split_idx == 12
    assert t_test[0] == t_train[-1]
    assert np.allclose(x_test[0], x_train[-1])

    rhs = make_predictor_rhs([LinearDecayPredictor(0.4)])
    pred = integrate_rhs(rhs, x[0], t)
    assert pred is not None
    assert safe_mse(x, pred) < 1e-8
    print("SUCCESS: integrated baseline helper tracks exponential decay")


if __name__ == "__main__":
    main()
