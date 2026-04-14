"""
Author: Chase Dunaway

Spacecraft Object for the ARGUS Satellite

The config.yaml file contains the physical properties of the satellite

References:
[1] F. L. Markley and J. L. Crassidis, Fundamentals of Spacecraft Attitude Determination and Control, ser. Space Technology Library. New
    York, NY: Springer, 2014, vol. 33.
"""


from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import yaml

from world.models.constants import MU_EARTH

class Spacecraft:
    """Spacecraft object for the ARGUS Satellite"""

    BASE_STATE_SIZE = 13 # [position(3), velocity(3), quaternion(4), omega(3)]

    def __init__(self, config_path: str | Path = Path(__file__).with_name("config.yaml")) -> None:
        """Initialize the spacecraft object from a YAML config file."""
        self.config_path = Path(config_path)

        self.names: list[str] = []
        self.debug: bool = False
        self.mass_vector: np.ndarray = np.empty((0,), dtype=float)
        self.inertia_tensor: np.ndarray = np.zeros((3, 3), dtype=float)
        self.position_vectors: np.ndarray = np.empty((0, 3), dtype=float)
        self.dimension_vectors: np.ndarray = np.empty((0, 3), dtype=float)
        self.face_dimensions: list[Dict[str, np.ndarray]] = []
        self.face_normals: list[Dict[str, np.ndarray]] = []

        # Configurable state, likely to be expanded later in HWs
        self.state_size: int = self.BASE_STATE_SIZE
        self.Idx: dict[str, dict[str, slice]] = {
            "X": {
                "POS_ECI": slice(0, 3),
                "VEL_ECI": slice(3, 6),
                "ATTITUDE": slice(6, 10),
                "ATTITUDE_RATE": slice(10, 13),
            }
        }
        self.state: np.ndarray = np.zeros(self.state_size, dtype=float)

        # Placeholder defaults; load_config sets these from YAML initial_conditions.
        self.position_eci: np.ndarray = np.zeros(3, dtype=float)
        self.velocity_eci: np.ndarray = np.zeros(3, dtype=float)
        self.attitude: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.attitude_rate: np.ndarray = np.zeros(3, dtype=float)

        self.load_config()

    @staticmethod
    def _property_value(items: Iterable[Dict], target_name: str, default: object) -> object:
        """Extract a named scalar value from a list of {name, value} property dictionaries."""
        for item in items:
            if str(item.get("name", "")).strip() == target_name:
                return item.get("value", default)
        return default

    @staticmethod
    def _as_bool(value: object, default: bool = False) -> bool:
        """Normalize config values into booleans."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off", ""}:
                return False
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    @classmethod
    def _property_bool(cls, items: Iterable[Dict], target_name: str, default: bool = False) -> bool:
        """Extract and normalize a named boolean property."""
        return cls._as_bool(cls._property_value(items, target_name, default), default)


    # Bunch of YAML parsing helpers
    @staticmethod
    def _geometric_center_vector(geometric_center: Dict | None) -> np.ndarray:
        """Build a 3D vector from geometric center values (x, y, z)."""
        geometric_center = geometric_center or {}
        return np.array(
            [
                float(geometric_center.get("x", 0.0)),
                float(geometric_center.get("y", 0.0)),
                float(geometric_center.get("z", 0.0)),
            ],
            dtype=float,
        )

    @staticmethod
    def _dimensions_vector(dimensions: Dict | None) -> np.ndarray:
        """Build a component size vector from YAML dimensions (x, y, z)."""
        dimensions = dimensions or {}
        return np.array(
            [
                float(dimensions.get("x", 0.0)),
                float(dimensions.get("y", 0.0)),
                float(dimensions.get("z", 0.0)),
            ],
            dtype=float,
        )

    @staticmethod
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
    
    def load_config(self) -> None:
        """Load physical properties from the spacecraft YAML config file."""
        with self.config_path.open("r", encoding="utf-8") as file:
            cfg = yaml.safe_load(file) or {}

        simulation_properties: Iterable[Dict] = cfg.get("simulation_properties", [])
        self.debug = self._property_bool(
            simulation_properties,
            "debug",
            default=self._as_bool(cfg.get("debug"), False),
        )

        dynamics_properties: Iterable[Dict] = cfg.get("dynamics_properties", [])
        raw_state_size = self._property_value(dynamics_properties, "state_size", self.BASE_STATE_SIZE)
        self.state_size = int(raw_state_size)
        if self.state_size < self.BASE_STATE_SIZE:
            raise ValueError(f"state_size must be >= {self.BASE_STATE_SIZE} to hold [r, v, q, omega]")

        self.Idx = {
            "X": {
                "POS_ECI": slice(0, 3),
                "VEL_ECI": slice(3, 6),
                "ATTITUDE": slice(6, 10),
                "ATTITUDE_RATE": slice(10, 13),
            }
        }
        self.state = np.zeros(self.state_size, dtype=float)

        physical_properties: Iterable[Dict] = cfg.get("physical_properties", [])
        initial_conditions: Iterable[Dict] = cfg.get("initial_conditions", [])

        position_init, velocity_init = self._state_from_orbital_elements(
            semi_major_axis_m=float(self._property_value(initial_conditions, "semi-major_axis", 6_968_000.0)),
            eccentricity=float(self._property_value(initial_conditions, "eccentricity", 0.0)),
            inclination_deg=float(self._property_value(initial_conditions, "inclination", 0.0)),
            raan_deg=float(self._property_value(initial_conditions, "raan", 0.0)),
            argument_of_perigee_deg=float(self._property_value(initial_conditions, "argument_of_perigee", 0.0)),
            true_anomaly_deg=float(self._property_value(initial_conditions, "true_anomaly", 0.0)),
        )

        attitude_init = np.asarray(
            self._property_value(initial_conditions, "attitude", [1.0, 0.0, 0.0, 0.0]),
            dtype=float,
        )
        attitude_rate_init = np.asarray(
            self._property_value(initial_conditions, "attitude_rate", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        if attitude_rate_init.ndim != 1 or attitude_rate_init.size != 3:
            raise ValueError("initial_conditions.attitude_rate must be a 3-element angular velocity vector [wx, wy, wz]")

        initial_state = np.zeros(self.state_size, dtype=float)
        initial_state[self.Idx["X"]["POS_ECI"]] = position_init
        initial_state[self.Idx["X"]["VEL_ECI"]] = velocity_init
        initial_state[self.Idx["X"]["ATTITUDE"]] = attitude_init
        initial_state[self.Idx["X"]["ATTITUDE_RATE"]] = attitude_rate_init
        self.set_state(initial_state)

        names: list[str] = []
        masses: list[float] = []
        dimension_vectors: list[np.ndarray] = []
        face_dimensions: list[Dict[str, np.ndarray]] = []
        face_normals: list[Dict[str, np.ndarray]] = []
        geometric_centers: list[np.ndarray] = []

        for item in physical_properties:
            name = str(item.get("name", "unknown_component"))
            mass = float(item.get("mass", 0.0))
            dimensions = self._dimensions_vector(item.get("dimensions"))
            geometric_center = self._geometric_center_vector(item.get("geometric_center"))
            component_faces = self._face_vectors(item.get("faces"))

            names.append(name)
            masses.append(mass)
            dimension_vectors.append(dimensions)
            face_dimensions.append(component_faces["face_dimensions"])
            face_normals.append(component_faces["face_normals"])
            geometric_centers.append(geometric_center)

        self.names = names
        self.mass_vector = np.asarray(masses, dtype=float)
        self.dimension_vectors = (
            np.vstack(dimension_vectors) if dimension_vectors else np.empty((0, 3), dtype=float)
        )
        self.position_vectors = (
            np.vstack(geometric_centers) if geometric_centers else np.empty((0, 3), dtype=float)
        )
        self.face_dimensions = face_dimensions
        self.face_normals = face_normals
        self.compute_inertia_tensor()

    @staticmethod
    def _state_from_orbital_elements(
        semi_major_axis_m: float,
        eccentricity: float,
        inclination_deg: float,
        raan_deg: float,
        argument_of_perigee_deg: float,
        true_anomaly_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert classical orbital elements to Cartesian position and velocity."""
        a = float(semi_major_axis_m)
        e = float(eccentricity)
        inc = np.deg2rad(float(inclination_deg))
        raan = np.deg2rad(float(raan_deg))
        argp = np.deg2rad(float(argument_of_perigee_deg))
        nu = np.deg2rad(float(true_anomaly_deg))

        if e < 0.0 or e >= 1.0:
            raise ValueError("0 <= e < 1")

        p = a * (1.0 - e * e) # Eq 10.15 [1]
        cos_nu = np.cos(nu)
        sin_nu = np.sin(nu)
        r_norm = p / (1.0 + e * cos_nu) # Eq 10.21 [1]

        r_pqw = np.array([r_norm * cos_nu, r_norm * sin_nu, 0.0], dtype=float)  # Eq 10.49a [1] (inclination is in R)
        v_pqw = np.sqrt(MU_EARTH / p) * np.array([-sin_nu, e + cos_nu, 0.0], dtype=float) # Eq 10.49b [1] (inclination is in R)

        c_raan, s_raan = np.cos(raan), np.sin(raan)
        c_inc, s_inc = np.cos(inc), np.sin(inc)
        c_argp, s_argp = np.cos(argp), np.sin(argp)

        rotation = np.array(
            [
                [c_raan * c_argp - s_raan * s_argp * c_inc, -c_raan * s_argp - s_raan * c_argp * c_inc, s_raan * s_inc],
                [s_raan * c_argp + c_raan * s_argp * c_inc, -s_raan * s_argp + c_raan * c_argp * c_inc, -c_raan * s_inc],
                [s_argp * s_inc, c_argp * s_inc, c_inc],
            ],
            dtype=float,
        ) # Eq 10.75 [1]

        position = rotation @ r_pqw
        velocity = rotation @ v_pqw
        return position, velocity
    
    def set_state(self, state: np.ndarray) -> None:
        """Set the full spacecraft state vector."""
        if state.ndim != 1 or state.size != self.state_size:
            raise ValueError(f"state must be a 1D vector with {self.state_size} elements")

        self.state = state.astype(float)
        self.position_eci = self.state[self.Idx["X"]["POS_ECI"]]
        self.velocity_eci = self.state[self.Idx["X"]["VEL_ECI"]]
        self.attitude = self.state[self.Idx["X"]["ATTITUDE"]]
        self.attitude_rate = self.state[self.Idx["X"]["ATTITUDE_RATE"]]

    def get_state(self) -> np.ndarray:
        """Return full state vector sized by the model configuration."""
        self.state[self.Idx["X"]["POS_ECI"]] = self.position_eci
        self.state[self.Idx["X"]["VEL_ECI"]] = self.velocity_eci
        self.state[self.Idx["X"]["ATTITUDE"]] = self.attitude
        self.state[self.Idx["X"]["ATTITUDE_RATE"]] = self.attitude_rate
        return self.state

    def compute_center_of_mass(self) -> np.ndarray:
        """Compute the spacecraft center of mass from loaded component data."""
        if self.mass_vector.size == 0:
            raise ValueError("No component masses loaded from config")

        total_mass = float(np.sum(self.mass_vector))
        if total_mass == 0.0:
            raise ValueError("Total mass is zero")

        com = np.sum(self.mass_vector[:, np.newaxis] * self.position_vectors, axis=0) / total_mass

        return com

    def compute_inertia_tensor(self) -> np.ndarray:
        """Compute inertia tensor from loaded component masses and positions."""

        center_of_mass = self.compute_center_of_mass()
        inertia_tensor = np.zeros((3, 3), dtype=float)
        component_inertias: list[tuple[str, np.ndarray]] = []

        for name, mass, position, dimensions in zip(
            self.names,
            self.mass_vector,
            self.position_vectors,
            self.dimension_vectors,
        ):
            dx, dy, dz = np.asarray(dimensions, dtype=float)
            local_inertia = (float(mass) / 12.0) * np.diag( # Inertia for rectangular prisms
                [
                    dy**2 + dz**2,
                    dx**2 + dz**2,
                    dx**2 + dy**2,
                ]
            )

            r = np.asarray(position, dtype=float) - center_of_mass
            parallel_axis = float(mass) * (np.dot(r, r) * np.eye(3) - np.outer(r, r    ))
            component_inertia = local_inertia + parallel_axis
            component_inertias.append((name, component_inertia))
            inertia_tensor += component_inertia

        self.inertia_tensor = inertia_tensor
        for name, component_inertia in component_inertias:
            print(f"{name} inertia contribution [kg m^2]:")
            print(np.array2string(component_inertia, precision=6, separator=", "))
        print("Spacecraft inertia tensor [kg m^2]:")
        print(np.array2string(self.inertia_tensor, precision=6, separator=", "))

        return self.inertia_tensor
        