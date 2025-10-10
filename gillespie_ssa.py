#!/usr/bin/env python3
"""
Gillespie Stochastic Simulation Algorithm (SSA)

- Dependency-free, pure Python implementation
- Supports mass-action reactions of arbitrary order
- Includes runnable demos: SIR model and birth-death process

Usage examples:
  python gillespie_ssa.py --demo sir --S0 990 --I0 10 --R0 0 --beta 0.3 --gamma 0.1 --tmax 160 --seed 1
  python gillespie_ssa.py --demo birth-death --X0 50 --birth 0.6 --death 0.5 --tmax 50 --seed 2

You can optionally write a CSV with --csv path.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


State = Dict[str, int]


@dataclass(frozen=True)
class Reaction:
    """Mass-action reaction definition.

    Example: S + I -> 2I at rate beta / N
      reactants = {"S": 1, "I": 1}
      products  = {"I": 2}
      rate_constant = beta / N

    The propensity is computed as:
      a = rate_constant * Π_i C(n_i, r_i)
    where C(n, k) is the binomial coefficient and r_i are reactant stoichiometries.
    """

    name: str
    reactants: Dict[str, int]
    products: Dict[str, int]
    rate_constant: float

    def propensity(self, state: State) -> float:
        h = 1.0
        for species, required in self.reactants.items():
            available = state.get(species, 0)
            if available < required:
                return 0.0
            # math.comb handles large values efficiently and exactly
            h *= math.comb(available, required)
        return self.rate_constant * h

    def apply(self, state: State) -> None:
        # Consume reactants
        for species, amount in self.reactants.items():
            state[species] = state.get(species, 0) - amount
        # Produce products
        for species, amount in self.products.items():
            state[species] = state.get(species, 0) + amount
        # Ensure no negatives due to misuse
        for species, count in list(state.items()):
            if count < 0:
                raise ValueError(f"Negative molecules for species '{species}': {count}")


def gillespie_ssa(
    initial_state: State,
    reactions: Sequence[Reaction],
    t_max: float,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    record_initial: bool = True,
) -> Tuple[List[float], List[State]]:
    """Run Gillespie SSA for mass-action reactions.

    Returns lists of event times and corresponding states (copies).
    The state list aligns with the times list. If record_initial is True,
    the first entry is (t=0, initial_state).
    """
    if t_max <= 0:
        raise ValueError("t_max must be positive")

    rng = random.Random(seed)

    time_points: List[float] = []
    state_history: List[State] = []

    time_now: float = 0.0
    state_now: State = {k: int(v) for k, v in initial_state.items()}

    if record_initial:
        time_points.append(time_now)
        state_history.append(state_now.copy())

    steps: int = 0

    while time_now < t_max:
        if max_steps is not None and steps >= max_steps:
            break

        propensities = [rxn.propensity(state_now) for rxn in reactions]
        a0 = sum(propensities)
        if a0 <= 0.0:
            # No possible reactions; system is stuck
            break

        # Draw time increment tau ~ Exp(a0)
        u1 = rng.random()
        while u1 <= 0.0:
            u1 = rng.random()
        tau = -math.log(u1) / a0

        # Select reaction index with probability proportional to propensity
        threshold = rng.random() * a0
        cumulative = 0.0
        chosen_index = 0
        for i, a in enumerate(propensities):
            cumulative += a
            if cumulative >= threshold:
                chosen_index = i
                break

        time_now += tau
        if time_now > t_max:
            # Do not record states beyond t_max
            break

        # Apply chosen reaction
        reactions[chosen_index].apply(state_now)

        time_points.append(time_now)
        state_history.append(state_now.copy())

        steps += 1

    return time_points, state_history


def save_trajectory_csv(
    path: str,
    times: Sequence[float],
    states: Sequence[State],
    species_order: Sequence[str],
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", *species_order])
        for t, s in zip(times, states):
            writer.writerow([f"{t:.9f}", *[s.get(sp, 0) for sp in species_order]])


# ------------------------------ Demos ---------------------------------

def run_demo_sir(
    S0: int,
    I0: int,
    R0: int,
    beta: float,
    gamma: float,
    t_max: float,
    seed: Optional[int],
    csv_path: Optional[str],
) -> None:
    species = ["S", "I", "R"]
    N = S0 + I0 + R0
    if N <= 0:
        raise ValueError("Total population must be positive")

    reactions = [
        # Infection: S + I -> 2I with rate beta / N
        Reaction(
            name="infection",
            reactants={"S": 1, "I": 1},
            products={"I": 2},
            rate_constant=beta / float(N),
        ),
        # Recovery: I -> R with rate gamma
        Reaction(
            name="recovery",
            reactants={"I": 1},
            products={"R": 1},
            rate_constant=gamma,
        ),
    ]

    initial_state = {"S": S0, "I": I0, "R": R0}
    times, states = gillespie_ssa(initial_state, reactions, t_max=t_max, seed=seed)

    final_state = states[-1] if states else initial_state
    print("SIR demo complete")
    print(f"Events: {max(0, len(times) - 1)}  Duration: {times[-1] if times else 0:.3f}")
    print(f"Final counts: S={final_state.get('S',0)}  I={final_state.get('I',0)}  R={final_state.get('R',0)}")

    if csv_path:
        save_trajectory_csv(csv_path, times, states, species)
        print(f"Wrote trajectory CSV: {csv_path}")


def run_demo_birth_death(
    X0: int,
    birth_rate: float,
    death_rate: float,
    t_max: float,
    seed: Optional[int],
    csv_path: Optional[str],
) -> None:
    species = ["X"]
    reactions = [
        # Birth: X -> 2X at rate b per individual
        Reaction(
            name="birth",
            reactants={"X": 1},
            products={"X": 2},
            rate_constant=birth_rate,
        ),
        # Death: X -> 0 at rate d per individual
        Reaction(
            name="death",
            reactants={"X": 1},
            products={},
            rate_constant=death_rate,
        ),
    ]

    initial_state = {"X": X0}
    times, states = gillespie_ssa(initial_state, reactions, t_max=t_max, seed=seed)

    final_state = states[-1] if states else initial_state
    print("Birth-Death demo complete")
    print(f"Events: {max(0, len(times) - 1)}  Duration: {times[-1] if times else 0:.3f}")
    print(f"Final counts: X={final_state.get('X',0)}")

    if csv_path:
        save_trajectory_csv(csv_path, times, states, species)
        print(f"Wrote trajectory CSV: {csv_path}")


# ------------------------------ CLI -----------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gillespie SSA demos")
    subparsers = parser.add_subparsers(dest="demo", required=True)

    sir = subparsers.add_parser("sir", help="Run SIR model demo")
    sir.add_argument("--S0", type=int, default=990, help="Initial S")
    sir.add_argument("--I0", type=int, default=10, help="Initial I")
    sir.add_argument("--R0", type=int, default=0, help="Initial R")
    sir.add_argument("--beta", type=float, default=0.3, help="Infection rate beta")
    sir.add_argument("--gamma", type=float, default=0.1, help="Recovery rate gamma")
    sir.add_argument("--tmax", type=float, default=160.0, help="Simulation end time")
    sir.add_argument("--seed", type=int, default=None, help="Random seed")
    sir.add_argument("--csv", type=str, default=None, help="Optional CSV output path")

    bd = subparsers.add_parser("birth-death", help="Run birth-death demo")
    bd.add_argument("--X0", type=int, default=50, help="Initial X")
    bd.add_argument("--birth", type=float, default=0.6, help="Birth rate b")
    bd.add_argument("--death", type=float, default=0.5, help="Death rate d")
    bd.add_argument("--tmax", type=float, default=50.0, help="Simulation end time")
    bd.add_argument("--seed", type=int, default=None, help="Random seed")
    bd.add_argument("--csv", type=str, default=None, help="Optional CSV output path")

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    if args.demo == "sir":
        run_demo_sir(
            S0=args.S0,
            I0=args.I0,
            R0=args.R0,
            beta=args.beta,
            gamma=args.gamma,
            t_max=args.tmax,
            seed=args.seed,
            csv_path=args.csv,
        )
    elif args.demo == "birth-death":
        run_demo_birth_death(
            X0=args.X0,
            birth_rate=args.birth,
            death_rate=args.death,
            t_max=args.tmax,
            seed=args.seed,
            csv_path=args.csv,
        )
    else:
        raise ValueError(f"Unknown demo: {args.demo}")


if __name__ == "__main__":
    main()
