"""
Author: Chase Dunaway

Runs the simulation of the ARGUS Satellite
"""

from __future__ import annotations

from pathlib import Path
import yaml
	
from simulator import Simulator


def main() -> None:
	config_path = Path(__file__).with_name("config.yaml")
	with config_path.open("r", encoding="utf-8") as file:
		cfg = yaml.safe_load(file) or {}

	monte_carlo = False
	for item in cfg.get("simulation_properties", []) or []:
		if str(item.get("name", "")).strip() == "monte_carlo":
			monte_carlo = bool(item.get("value", False))
			break

	sim = Simulator(config_path=config_path)

	if monte_carlo:
		summary = sim.run_monte_carlo()
		sim.plot_monte_carlo_trials(summary, show=False)
		print("Monte Carlo complete")
		print(f"Root directory: {summary['root_dir']}")
		print(f"Trials: {summary['trials']}")
		print(f"Completed: {summary['completed']}")
		return

	result = sim.run()
	sim.plot_simulation(result, show=False)
	if sim.show_momentum_sphere_plot:
		sim.plot_momentum_sphere(result, show=False)

	print("Simulation complete")
	print(f"Orbit period: {result['orbit_period_s']:.2f} s")
	print(f"Duration: {result['sim_duration_s']:.2f} s")
	print(f"Steps: {result['num_steps']}")
	print(f"Log file: {result['log_file']}")
	print("Final spacecraft state (SI units):")
	print(f"  position [m]: {sim.spacecraft.position_eci}")
	print(f"  velocity [m/s]: {sim.spacecraft.velocity_eci}")
	print(f"  attitude [-]: {sim.spacecraft.attitude}")
	print(f"  omega [rad/s]: {sim.spacecraft.attitude_rate}")
	print(f"  rho [kg m^2/s]: {sim.spacecraft.rho}")


if __name__ == "__main__":
	main()
