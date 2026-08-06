import argparse
import numpy as np
import os
import json
import signal
from contextlib import contextmanager
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp

DOMAINS = ["ecology", "real_nile", "real_sunspots"]


def method_status(*values):
    return "ok" if all(np.isfinite(v) for v in values if v is not None) else "diverged"


def json_ready(value):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@contextmanager
def timeout_after(seconds):
    if seconds is None or seconds <= 0:
        yield
        return

    def _handle_timeout(_signum, _frame):
        raise TimeoutError(f"operation exceeded {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def load_pysindy():
    try:
        import pysindy as ps
    except ImportError:
        return None
    return ps


def load_pysr_regressor():
    try:
        from pysr import PySRRegressor
    except ImportError:
        return None
    return PySRRegressor


def safe_mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        return float("inf")
    if not np.all(np.isfinite(y_pred)):
        return float("inf")
    return float(mean_squared_error(y_true, y_pred))


def split_train_test(t, X, train_fraction=0.8):
    split_idx = int(len(t) * train_fraction)
    split_idx = min(max(split_idx, 2), len(t) - 1)
    return t[:split_idx], X[:split_idx], t[split_idx - 1:], X[split_idx - 1:], split_idx


def integrate_rhs(rhs, y0, t_eval, timeout_seconds=60):
    if len(t_eval) < 2:
        return np.asarray(y0, dtype=float)[None, :]

    try:
        max_step = max(float(np.median(np.diff(t_eval))), 1e-6)
        with timeout_after(timeout_seconds):
            sol = solve_ivp(
                rhs,
                (float(t_eval[0]), float(t_eval[-1])),
                np.asarray(y0, dtype=float),
                t_eval=np.asarray(t_eval, dtype=float),
                method="LSODA",
                rtol=1e-6,
                atol=1e-8,
                max_step=max_step,
            )
    except TimeoutError as e:
        print(f"Trajectory integration timed out: {e}")
        return None
    except Exception as e:
        print(f"Trajectory integration failed: {e}")
        return None

    if not sol.success or sol.y.shape[1] != len(t_eval):
        print(f"Trajectory integration failed: {sol.message}")
        return None

    trajectory = sol.y.T
    if not np.all(np.isfinite(trajectory)):
        print("Trajectory integration produced non-finite values.")
        return None
    return trajectory


def make_predictor_rhs(models):
    def rhs(_t, y):
        state = np.asarray(y, dtype=float).reshape(1, -1)
        derivs = []
        for model in models:
            pred = np.asarray(model.predict(state), dtype=float).reshape(-1)
            derivs.append(float(pred[0]))
        derivs = np.asarray(derivs, dtype=float)
        if not np.all(np.isfinite(derivs)):
            raise FloatingPointError("non-finite derivative prediction")
        return derivs

    return rhs

def load_data(domain):
    data_path = f"data/{domain}_ground_truth.npy"
    time_path = f"data/{domain}_time_points.npy"
    if not os.path.exists(data_path) or not os.path.exists(time_path):
        # Trigger generator if missing
        if domain == "ecology":
            import ecology_data_loader as dl
            dl.generate_data()
        elif domain in {"real_nile", "real_sunspots"}:
            import build_real_data  # noqa: F401 - import writes the public real-data arrays
        else:
            raise FileNotFoundError(f"No data generator configured for {domain}")
    
    X = np.load(data_path)
    t = np.load(time_path)
    return t, X

def run_sindy(t, X):
    ps = load_pysindy()
    if ps is None:
        print("PySINDy not installed.")
        return {"status": "missing_dependency", "train_mse": None, "test_mse": None, "complexity": None}
    
    # Train-test split (80% train, 20% test for extrapolation)
    t_train, X_train, t_test, X_test, split_idx = split_train_test(t, X)
    
    # Fit SINDy model
    model = ps.SINDy()
    model.fit(X_train, t=t_train)
    
    # Print discovered equations
    print("Discovered SINDy Equations:")
    names = [f"y{i}" for i in range(X.shape[1])]
    model.print(lhs=names)
    
    # Train MSE
    try:
        X_train_pred = model.simulate(X_train[0], t_train)
        train_mse = safe_mse(X_train[1:], X_train_pred[1:])
    except Exception as e:
        print(f"SINDy simulation diverged on train set: {e}")
        train_mse = float("inf")
    
    # Extrapolation (Test) MSE
    try:
        X_test_pred = model.simulate(X_test[0], t_test)
        test_mse = safe_mse(X_test[1:], X_test_pred[1:])
    except Exception as e:
        print(f"SINDy simulation diverged on test set: {e}")
        test_mse = float("inf")
        
    # Complexity: count non-zero coefficients
    complexity = np.count_nonzero(model.coefficients())
    
    return {
        "status": method_status(train_mse, test_mse),
        "metric": "integrated_trajectory_mse",
        "split_index": split_idx,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "complexity": int(complexity),
    }

def run_pysr(t, X, niterations=20, integration_timeout=60):
    PySRRegressor = load_pysr_regressor()
    if PySRRegressor is None:
        print("PySR not installed.")
        return {"status": "missing_dependency", "train_mse": None, "test_mse": None, "complexity": None}
    
    t_train, X_train, t_test, X_test, split_idx = split_train_test(t, X)
    
    # Estimate derivatives numerically for PySR
    # dy/dt = f(y)
    dX_dt = np.zeros_like(X_train)
    for i in range(X_train.shape[1]):
        dX_dt[:, i] = np.gradient(X_train[:, i], t_train)
        
    # Run PySR Regressor for each state variable.
    # PySR is fit to local numerical derivatives here, so we report derivative
    # prediction error rather than pretending we have a stable integrated ODE.
    equations = []
    complexities = []
    train_deriv_mse = 0.0
    test_deriv_mse = 0.0
    models = []
    
    for i in range(X_train.shape[1]):
        print(f"Running PySR for dimension {i}...")
        model = PySRRegressor(
            niterations=niterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp"],
            model_selection="best",
            output_directory="artifacts/baselines/pysr_outputs",
        )
        model.fit(X_train, dX_dt[:, i])
        best = model.get_best()
        try:
            equation = best["equation"]
            complexity = best["complexity"]
        except (KeyError, TypeError):
            equation = best.equation
            complexity = best.complexity
        equations.append(str(equation))
        complexities.append(int(complexity))
        models.append(model)
        train_pred = model.predict(X_train)
        train_deriv_mse += safe_mse(dX_dt[:, i], train_pred)

        dX_test_dt = np.gradient(X_test[:, i], t_test)
        try:
            test_pred = model.predict(X_test)
            test_deriv_mse += safe_mse(dX_test_dt, test_pred)
        except Exception as e:
            print(f"PySR derivative prediction failed on test set: {e}")
            test_deriv_mse = float("inf")
        
    print(f"Discovered PySR Equations: {equations}")

    rhs = make_predictor_rhs(models)
    train_traj = integrate_rhs(rhs, X_train[0], t_train, timeout_seconds=integration_timeout)
    if train_traj is None:
        train_mse = float("inf")
    else:
        train_mse = safe_mse(X_train[1:], train_traj[1:])

    test_traj = integrate_rhs(rhs, X_test[0], t_test, timeout_seconds=integration_timeout)
    if test_traj is None:
        test_mse = float("inf")
    else:
        test_mse = safe_mse(X_test[1:], test_traj[1:])

    return {
        "status": method_status(train_mse, test_mse),
        "metric": "integrated_trajectory_mse",
        "split_index": split_idx,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "complexity": int(sum(complexities)),
        "equations": equations,
        "derivative_train_mse": float(train_deriv_mse),
        "derivative_test_mse": float(test_deriv_mse),
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Run saved baseline comparisons.")
    parser.add_argument("--domains", nargs="+", default=DOMAINS)
    parser.add_argument("--pysr-iterations", type=int, default=20)
    parser.add_argument("--integration-timeout", type=int, default=60)
    parser.add_argument("--output", default="artifacts/baselines/baseline_results.json")
    parser.add_argument("--resume", action="store_true", help="Skip domains already present in output JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=== Running Baseline Comparison Benchmarks ===")
    out_path = args.output
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.resume and os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
    else:
        results = {}
    os.makedirs("artifacts/baselines", exist_ok=True)
    for domain in args.domains:
        if args.resume and domain in results:
            print(f"\nSkipping Domain: {domain} (already in {out_path})")
            continue

        print(f"\nEvaluating Domain: {domain}")
        t, X = load_data(domain)
        
        print("\n--- Running SINDy ---")
        sindy_result = run_sindy(t, X)
        
        print("\n--- Running PySR ---")
        pysr_result = run_pysr(t, X, niterations=args.pysr_iterations, integration_timeout=args.integration_timeout)
        
        results[domain] = {
            "SINDy": sindy_result,
            "PySR": pysr_result,
        }
        with open(out_path, "w") as f:
            json.dump(json_ready(results), f, indent=2, sort_keys=True, allow_nan=False)

    print("\n=== FINAL RESULTS ===")
    print(results)
    with open(out_path, "w") as f:
        json.dump(json_ready(results), f, indent=2, sort_keys=True, allow_nan=False)
    print(f"Saved baseline results to {out_path}")

if __name__ == "__main__":
    main()
