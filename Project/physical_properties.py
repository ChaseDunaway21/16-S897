"""
Author: Chase Dunaway

Calculate the center of mass, inertia tensor, and principal axes of the ARGUS Satellite

The config.yaml file contains the physical properties of the satellite

OUTPUTS: 
	mass vector, 
	geometric center vectors, 
	discretized face dimensions, 
	discretized face normals, 
	component names
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _geometric_center_vector(geometric_center: Dict | None) -> np.ndarray:
	"""Build a 3D vector from YAML geometric center values (x, y, z)."""
	geometric_center = geometric_center or {}
	return np.array(
		[
			float(geometric_center.get("x", 0.0)),
			float(geometric_center.get("y", 0.0)),
			float(geometric_center.get("z", 0.0)),
		],
		dtype=float,
	)


def _face_vectors(face_config: Dict | None) -> Dict[str, Dict[str, np.ndarray]]:
	"""Build nested face-dimension and face-normal vectors from YAML face values."""
	face_config = face_config or {}
	face_dimensions = face_config.get("face_dimensions", {}) or {}
	face_normals = face_config.get("face_normals", {}) or {}

	return {
		"face_dimensions": {
			str(face_name): np.asarray(face_vector, dtype=float)
			for face_name, face_vector in face_dimensions.items()
		},
		"face_normals": {
			str(face_name): np.asarray(face_vector, dtype=float)
			for face_name, face_vector in face_normals.items()
		},
	}


def load_config_vectors(
	config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> Tuple[
	np.ndarray,
	np.ndarray,
	List[Dict[str, np.ndarray]],
	List[Dict[str, np.ndarray]],
	List[str],
]:
	"""Load mass and geometric center vectors from a YAML config file."""
	config_path = Path(config_path)
	with config_path.open("r", encoding="utf-8") as file:
		cfg = yaml.safe_load(file) or {}

	physical_properties: Iterable[Dict] = cfg.get("physical_properties", [])

	names: List[str] = []
	masses: List[float] = []
	face_dimensions: List[Dict[str, np.ndarray]] = []
	face_normals: List[Dict[str, np.ndarray]] = []
	geo_center: List[np.ndarray] = []

	for item in physical_properties:
		name = str(item.get("name", "unknown_component"))
		mass = float(item.get("mass", 0.0))
		geometric_center = _geometric_center_vector(item.get("geometric_center"))
		component_faces = _face_vectors(item.get("faces"))

		names.append(name)
		masses.append(mass)
		face_dimensions.append(component_faces["face_dimensions"])
		face_normals.append(component_faces["face_normals"])
		geo_center.append(geometric_center)

	mass_vector = np.asarray(masses, dtype=float)
	position_vectors = np.vstack(geo_center) if geo_center else np.empty((0, 3), dtype=float)
	return mass_vector, position_vectors, face_dimensions, face_normals, names


def compute_center_of_mass(mass_vector: np.ndarray, position_vectors: np.ndarray) -> np.ndarray:
    """Compute the center of mass"""
    total_mass = np.sum(mass_vector)
    center_of_mass = np.sum(mass_vector[:, np.newaxis] * position_vectors, axis=0) / total_mass
    return center_of_mass


def compute_inertia_tensor(mass_vector: np.ndarray, position_vectors: np.ndarray) -> np.ndarray:
    """Compute the inertia tensor"""
    inertia_tensor = np.zeros((3, 3), dtype=float)
    for mass, position in zip(mass_vector, position_vectors):
        r_squared = np.dot(position, position)
        inertia_tensor += mass * (r_squared * np.eye(3) - np.outer(position, position))
    return inertia_tensor


if __name__ == "__main__":
	mass_vector, position_vectors, face_dimensions, face_normals, names = load_config_vectors()
	center_of_mass = compute_center_of_mass(mass_vector, position_vectors)
	inertia_tensor = compute_inertia_tensor(mass_vector, position_vectors)
	print("Components:", names)
	print("Mass vector:", mass_vector)
	print("Face dimensions:\n", face_dimensions)
	print("Face normals:\n", face_normals)
	print("Geometric center vectors:\n", position_vectors)
	print("Center of Mass:\n", center_of_mass)
	print("Inertia Tensor:\n", inertia_tensor)
