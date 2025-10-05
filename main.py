"""
Gillespie Stochastic Simulation Algorithm (SSA) for a simple reaction:

    A  --c-->  B

This script simulates the time evolution of molecule counts for a
unimolecular conversion reaction A -> B with rate constant c using the
exact SSA (Gillespie, 1977).

Run:
    python3 main.py --A0 50 --rate 0.1 --tmax 50 --seed 42

All steps are explained with inline comments.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SimulationResult:
    """Container for SSA outputs for clarity."""
    times: List[float]
    counts_A: List[int]
    counts_B: List[int]


def simulate_gillespie_unimolecular_decay(
    initial_count_a: int,
    rate_constant: float,
    t_max: float,
    seed: int | None = None,
) -> SimulationResult:
    """
    Run Gillespie SSA for A -> B with propensity a1 = c * N_A.

    Args:
        initial_count_a: Initial number of A molecules (>= 0).
        rate_constant: Reaction rate constant c (> 0).
        t_max: Simulation horizon in the same time units as c (> 0).
        seed: Optional random seed for reproducibility.

    Returns:
        SimulationResult with arrays of times and counts for A and B.
    """
    # 1) Validate inputs early to fail-fast on incorrect usage
    if not isinstance(initial_count_a, int) or initial_count_a < 0:
        raise ValueError("initial_count_a must be a non-negative integer")
    if not (isinstance(rate_constant, (int, float)) and rate_constant > 0):
        raise ValueError("rate_constant must be a positive number")
    if not (isinstance(t_max, (int, float)) and t_max > 0):
        raise ValueError("t_max must be a positive number")

    # 2) Initialize RNG (random number generator)
    rng = random.Random(seed)

    # 3) Initialize simulation state and output storage
    time_now = 0.0
    count_a = int(initial_count_a)
    count_b = 0

    times: List[float] = [time_now]
    counts_a: List[int] = [count_a]
    counts_b: List[int] = [count_b]

    # 4) Main SSA loop: keep generating reaction events until we reach t_max
    while time_now < t_max:
        # 4a) Compute propensity (rate at which the reaction fires)
        #     For A -> B, the propensity is a1 = c * N_A because each A can decay
        propensity_a1 = rate_constant * count_a

        # 4b) If the total propensity is zero, no more reactions can occur
        if propensity_a1 <= 0.0:
            # No events possible (e.g., N_A == 0): stop the simulation
            break

        total_propensity = propensity_a1  # single-reaction case

        # 4c) Draw two independent uniform random numbers in (0, 1]
        #     Use tiny lower bound to avoid log(0)
        r1 = max(rng.random(), 1e-15)
        r2 = max(rng.random(), 1e-15)

        # 4d) Compute the time to the next reaction (exponential waiting time)
        #     τ = (1 / a0) * ln(1 / r1)
        tau = (1.0 / total_propensity) * math.log(1.0 / r1)

        # 4e) Determine which reaction fires next
        #     With a single reaction, it always fires if total_propensity > 0.
        #     In multi-reaction systems, we'd select by cumulative sums vs r2*a0.

        # 4f) If the next event would occur after t_max, stop before exceeding horizon
        if time_now + tau > t_max:
            # Optionally record the final time horizon (holding counts constant)
            time_now = t_max
            times.append(time_now)
            counts_a.append(count_a)
            counts_b.append(count_b)
            break

        # 4g) Advance time by tau (we jump directly to the next event)
        time_now += tau

        # 4h) Update the state according to the reaction stoichiometry: A -> B
        #     Consuming 1 A and producing 1 B
        count_a -= 1
        count_b += 1

        # 4i) Record the new state after the event
        times.append(time_now)
        counts_a.append(count_a)
        counts_b.append(count_b)

    return SimulationResult(times=times, counts_A=counts_a, counts_B=counts_b)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for convenience."""
    parser = argparse.ArgumentParser(
        description=(
            "Gillespie SSA for the unimolecular reaction A -> B (a1 = c * N_A)."
        )
    )
    parser.add_argument("--A0", type=int, default=50, help="Initial count of A (>= 0)")
    parser.add_argument(
        "--rate",
        type=float,
        default=0.1,
        help="Rate constant c (> 0)",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=50.0,
        help="Simulation horizon time (> 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = simulate_gillespie_unimolecular_decay(
        initial_count_a=args.A0,
        rate_constant=args.rate,
        t_max=args.tmax,
        seed=args.seed,
    )

    # Print a simple table of times and counts
    print("t\tA\tB")
    for t, a, b in zip(result.times, result.counts_A, result.counts_B):
        print(f"{t:.6f}\t{a}\t{b}")


if __name__ == "__main__":
    main()
