"""
Author: Chase Dunaway

Runs the simulation of the ARGUS Satellite
"""

from __future__ import annotations

from pathlib import Path

from simulator import Simulator


def main() -> None:
	config_path = Path(__file__).with_name("config.yaml")
	sim = Simulator(config_path=config_path)
	result = sim.run()
	sim.plot_simulation(result, show=False)
	sim.plot_momentum_sphere(result, show=False)

	print("Simulation complete")
	print(f"Orbit period: {result['orbit_period_s']:.2f} s")
	print(f"Duration: {result['sim_duration_s']:.2f} s")
	print(f"Steps: {result['num_steps']}")
	print(f"Log file: {result['log_file']}")
	print("Final spacecraft state (SI units):")
	print(f"  position [m]: {sim.spacecraft.position_ecef}")
	print(f"  velocity [m/s]: {sim.spacecraft.velocity_ecef}")
	print(f"  attitude [-]: {sim.spacecraft.attitude}")
	print(f"  omega [rad/s]: {sim.spacecraft.attitude_rate}")


if __name__ == "__main__":
	main()
