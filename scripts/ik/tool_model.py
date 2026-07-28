"""Authoritative fixed-tool model derived from the SO-101 URDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .kinematics_utils import (
    make_transform,
    require_transform,
    rotation_matrix_rpy,
)


@dataclass(frozen=True)
class FixedToolModel:
    """A fixed child tool frame expressed relative to its parent link."""

    parent_link: str
    tool_link: str
    joint_name: str
    parent_from_tool: np.ndarray

    def __post_init__(self) -> None:
        transform = require_transform(
            self.parent_from_tool,
            label="parent_from_tool",
        ).copy()
        transform.setflags(write=False)
        object.__setattr__(self, "parent_from_tool", transform)

    @property
    def position_in_parent_m(self) -> np.ndarray:
        return self.parent_from_tool[:3, 3].copy()

    @property
    def approach_axis_in_parent(self) -> np.ndarray:
        """Tool +Z, which points outward along the fixed SO-101 finger."""
        return self.parent_from_tool[:3, 2].copy()

    @property
    def closing_axis_in_parent(self) -> np.ndarray:
        """Tool -X, from the fixed fingertip toward the moving fingertip."""
        return -self.parent_from_tool[:3, 0].copy()


def fixed_tool_model_from_urdf(
    urdf_path: Path,
    *,
    joint_name: str,
    expected_parent_link: str,
    expected_tool_link: str,
) -> FixedToolModel:
    """Load and validate one fixed tool joint from a URDF file."""
    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Tool-model URDF does not exist: {path}")
    root = ET.parse(path).getroot()
    matching_joints = [
        joint
        for joint in root.findall("joint")
        if joint.attrib.get("name") == joint_name
    ]
    if len(matching_joints) != 1:
        raise ValueError(
            f"Expected exactly one URDF joint {joint_name!r}, "
            f"found {len(matching_joints)}"
        )
    joint = matching_joints[0]
    if joint.attrib.get("type") != "fixed":
        raise ValueError(f"Tool joint {joint_name!r} must be fixed")

    parent = joint.find("parent")
    child = joint.find("child")
    parent_link = parent.attrib.get("link") if parent is not None else None
    tool_link = child.attrib.get("link") if child is not None else None
    if parent_link != expected_parent_link or tool_link != expected_tool_link:
        raise ValueError(
            f"Tool joint {joint_name!r} must connect "
            f"{expected_parent_link!r} to {expected_tool_link!r}, got "
            f"{parent_link!r} to {tool_link!r}"
        )

    origin = joint.find("origin")
    xyz_text = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
    rpy_text = origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0"
    position = np.fromstring(xyz_text, sep=" ", dtype=np.float64)
    rpy = np.fromstring(rpy_text, sep=" ", dtype=np.float64)
    if position.shape != (3,) or rpy.shape != (3,):
        raise ValueError(
            f"Tool joint {joint_name!r} must have three-value xyz and rpy origins"
        )
    return FixedToolModel(
        parent_link=parent_link,
        tool_link=tool_link,
        joint_name=joint_name,
        parent_from_tool=make_transform(position, rotation_matrix_rpy(rpy)),
    )
