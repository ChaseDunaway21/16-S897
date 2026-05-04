"""Solve Wahba's problem from direct sensor getter calls at randomized poses.

Example bash:
python -m wahbas.wahbas_main --trials 100 --min-vectors 2 --plot --save ../results/wahba_monte_carlo.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import cyipopt as ipopt

from .wahbas_plotting import plot_wahba_attitude_trials, plot_wahba_monte_carlo
from .wahbas_sensor_gen import (
    generate_wahba_monte_carlo_samples,
    generate_wahba_sensor_sample,
)
from world.rotations_and_transformations import R_inertial_to_body
from world.math import unit_vector

#################################################################################################
# WAHBA SVD
#################################################################################################


def wahba_svd(body_vectors: np.ndarray, reference_vectors: np.ndarray) -> np.ndarray:
    # https://youtu.be/PhAwy3dEYBk?si=b_kLH5RPCbRHVhf3
    # This is discussed in class, but here is a summary:
    # "What if the bearing vectors are not linearly independent of each other?"
    # 1. B = sum(w_i * b_i * r_i^T) w = weight, b = body vector, r = reference vector
    #    The goal is to find min_Q || Q-B.T ||^2_F (Frobenius norm)
    #    The trick is the Frobenius norm, which can be written as a Trace:
    #        <A,B>_F = Tr(A.T B) = vec(A).T vec(B) <- Inner product of column vectorized matrices
    #        A = [A1, A2, A3, ...], vec (A) = [A1; A2; A3; ...]
    #        Thus,
    #        || A ||^2_F = <A, A>_F = Tr(A.T A) = vec(A).T vec(A)
    # 2. Rewrite the optimization problem using this trick:
    #      min_Q || Q-B.T ||^2_F
    #    = min_Q <Q-B.T, Q-B.T>_F
    #    = min_Q Tr[(Q-B.T)^T (Q-B.T)]
    #    = min_Q Tr[Q.T Q - 2 Q.T B.T + B B.T] <- Q.TQ and BB.T are constants that don't change min Q
    #    = min_Q -2 Tr[Q.T B.T] <- the constant 2 doesn't change, take out the negative to make it a max!
    #    = max_Q Tr[Q B]
    # 3. Replace Q with the SVD of B:
    #    B = U S V.T
    #    max Q Tr[Q B] = max Q Tr[Q U S V.T] = max Q <V Q.T U, S>_F <- Frobenius norm
    #        V.T and U have to be orthogonal, so the best solution is to make Q = I
    #    Thus,
    #    Q = U V.T

    unit_body_vectors = unit_vector(body_vectors)
    unit_reference_vectors = unit_vector(reference_vectors)

    B = body_vectors.T @ reference_vectors
    U, _, Vt = np.linalg.svd(B)
    M = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    return U @ M @ Vt


#################################################################################################
# WAHBA SDP
#################################################################################################
class WahbaIpoptProblem:
    def __init__(self, B: np.ndarray) -> None:
        self.B = B

    def objective(self, Q: np.ndarray) -> float:
        return -np.trace(Q.reshape(3, 3) @ self.B)

    def gradient(self, Q: np.ndarray) -> np.ndarray:
        return -self.B.T.reshape(-1)

    def constraints(self, Q: np.ndarray) -> np.ndarray:
        Q = Q.reshape(3, 3)
        return np.hstack(
            [
                (Q.T @ Q - np.eye(3)).flatten(),
                np.linalg.det(Q) - 1.0,
            ]
        )

    def jacobian(self, Q: np.ndarray) -> np.ndarray:
        Q = Q.reshape(3, 3)
        J = np.zeros((10, 9), dtype=float)

        # Constraint 1: Q.T Q = I
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    J[i * 3 + j, k * 3 + j] += Q[k, i]  # d/dQ[k,j] of Q.T Q[i,j]
                    J[i * 3 + j, k * 3 + i] += Q[k, j]  # d/dQ[k,i] of Q.T Q[i,j]

        # Constraint 2: det(Q) = 1
        cof = (
            np.linalg.det(Q) * np.linalg.inv(Q).T
        )  # The derivative of a determinant is just the cofactor
        J[9, :] = cof.flatten()

        return J.flatten()


def wahba_sdp(body_vectors: np.ndarray, reference_vectors: np.ndarray) -> np.ndarray:
    # https://youtu.be/PhAwy3dEYBk?si=b_kLH5RPCbRHVhf3
    # The optimization problem is the same as the SVD solution, but we add a constraint that Q is a valid rotation matrix
    #    max_Q Tr[Q B]
    #    s.t.
    #       I - Q.T Q = 0
    #       det(Q) = 1

    B = body_vectors.T @ reference_vectors

    # Optimization settings
    n = 9  # 3x3 rotation matrix
    m = 10  # 9 constraints for orthogonality, 1 constraint for determinant

    # Bounds
    lb = -np.inf * np.ones(n)  # No upper bound
    ub = np.inf * np.ones(n)  # No lower bound
    cl = np.zeros(m)  # Constraints lower bound is 0
    cu = np.zeros(m)  # Constraints upper bound is 0

    # Guess
    x0 = wahba_svd(body_vectors, reference_vectors).flatten()

    solver = ipopt.Problem(
        n=n,
        m=m,
        problem_obj=WahbaIpoptProblem(B),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )
    solver.add_option("print_level", 0)
    solver.add_option("tol", 1e-10)

    x_opt, info = solver.solve(x0)

    return x_opt.reshape(3, 3)


#################################################################################################
# HELPERS
#################################################################################################


def wahba_value(
    R_est: np.ndarray, body_vectors: np.ndarray, reference_vectors: np.ndarray
) -> float:
    B = body_vectors.T @ reference_vectors
    return float(np.trace(R_est.T @ B))


def rotation_error_deg(R_est: np.ndarray, R_true: np.ndarray) -> float:
    error = R_est @ R_true.T
    trace_arg = np.clip(0.5 * (np.trace(error) - 1.0), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(trace_arg)))


def solve_wahba_sample(sample: dict[str, object]) -> dict[str, object]:
    body_vectors = sample["body_vectors"]
    reference_vectors = sample["reference_vectors_eci"]
    R_svd = wahba_svd(body_vectors, reference_vectors)
    R_sdp = wahba_sdp(body_vectors, reference_vectors)
    R_true = R_inertial_to_body(sample["attitude_true"])
    return {
        "sensor_names": sample["sensor_names"],
        "body_vectors": body_vectors,
        "reference_vectors_eci": reference_vectors,
        "R_true": R_true,
        "R_svd": R_svd,
        "R_sdp": R_sdp,
        "svd_attitude_error_deg": rotation_error_deg(R_svd, R_true),
        "sdp_attitude_error_deg": rotation_error_deg(R_sdp, R_true),
        "svd_value": wahba_value(R_svd, body_vectors, reference_vectors),
        "sdp_value": wahba_value(R_sdp, body_vectors, reference_vectors),
    }


def attitude_plot_path(save_path: Path) -> Path:
    return save_path.with_name(f"{save_path.stem}_attitudes{save_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wahba from direct sensor getters")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time", type=float, default=0.0)
    parser.add_argument("--min-vectors", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--save",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "wahba_monte_carlo.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials > 1:
        samples = generate_wahba_monte_carlo_samples(
            config_path=args.config,
            seed=args.seed,
            time_s=args.time,
            min_vectors=args.min_vectors,
            max_attempts=args.max_attempts,
            trials=args.trials,
        )
        results = [solve_wahba_sample(sample) for sample in samples]
        attitude_errors_deg = np.array(
            [result["svd_attitude_error_deg"] for result in results],
            dtype=float,
        )
        sdp_errors_deg = np.array(
            [result["sdp_attitude_error_deg"] for result in results],
            dtype=float,
        )
        value_difference = np.array(
            [result["sdp_value"] - result["svd_value"] for result in results],
            dtype=float,
        )
        vector_counts = np.array(
            [len(result["sensor_names"]) for result in results],
            dtype=float,
        )

        print(f"Wahba Monte Carlo trials: {args.trials}")
        print(f"SVD mean attitude error: {np.mean(attitude_errors_deg):.6f} deg")
        print(f"SDP mean attitude error: {np.mean(sdp_errors_deg):.6f} deg")
        print(f"SVD max attitude error: {np.max(attitude_errors_deg):.6f} deg")
        print(f"SDP max attitude error: {np.max(sdp_errors_deg):.6f} deg")
        print(f"Mean value difference SDP-SVD: {np.mean(value_difference):.6e}")
        if args.plot or args.show:
            plot_wahba_monte_carlo(
                attitude_errors_deg,
                vector_counts,
                args.save,
                args.show,
            )
            attitudes_save = attitude_plot_path(args.save)
            plot_wahba_attitude_trials(results, attitudes_save, args.show)
            print(f"Wahba Monte Carlo plot: {args.save}")
            print(f"Wahba attitude trial plot: {attitudes_save}")
        return

    sample = generate_wahba_sensor_sample(
        config_path=args.config,
        seed=args.seed,
        time_s=args.time,
        min_vectors=args.min_vectors,
        max_attempts=args.max_attempts,
    )
    result = solve_wahba_sample(sample)

    print("Sensors:", ", ".join(result["sensor_names"]))
    print("Body vectors:")
    print(np.array2string(result["body_vectors"], precision=6, separator=", "))
    print("Reference vectors ECI:")
    print(np.array2string(result["reference_vectors_eci"], precision=6, separator=", "))
    print(f"SVD attitude error: {result['svd_attitude_error_deg']:.6f} deg")
    print(f"SDP attitude error: {result['sdp_attitude_error_deg']:.6f} deg")
    print(f"SVD value: {result['svd_value']:.12g}")
    print(f"SDP value: {result['sdp_value']:.12g}")
    print(f"Value difference SDP-SVD: {result['sdp_value'] - result['svd_value']:.6e}")

    if args.plot or args.show:
        plot_wahba_attitude_trials([result], args.save, args.show)
        print(f"Wahba attitude trial plot: {args.save}")


if __name__ == "__main__":
    main()
