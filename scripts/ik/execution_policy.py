"""Pure execution-policy helpers shared by simulation control code."""

from __future__ import annotations

from dataclasses import dataclass


def distributed_command_steps(total_steps: int, segment_count: int) -> list[int]:
    """Split a positive command budget evenly across path segments."""
    if total_steps < segment_count or segment_count <= 0:
        raise ValueError("Command budget must provide at least one step per segment")
    base_steps, remainder = divmod(total_steps, segment_count)
    return [
        base_steps + (1 if index < remainder else 0)
        for index in range(segment_count)
    ]


@dataclass
class DescentContactDiagnostics:
    """Record first contact events without making them insertion stop gates."""

    first_finger_contact_step: int | None = None
    first_finger_contact_segment: int | None = None
    first_two_finger_contact_step: int | None = None
    first_two_finger_contact_segment: int | None = None

    def observe(
        self,
        *,
        segment: int,
        step: int,
        fixed_contact: bool,
        moving_contact: bool,
    ) -> tuple[bool, bool]:
        """Record contact state and return first-any/first-two event flags."""
        if segment <= 0 or step <= 0:
            raise ValueError("Descent segment and step numbers must be positive")
        first_any = False
        first_two = False
        if (
            (fixed_contact or moving_contact)
            and self.first_finger_contact_step is None
        ):
            self.first_finger_contact_step = step
            self.first_finger_contact_segment = segment
            first_any = True
        if (
            fixed_contact
            and moving_contact
            and self.first_two_finger_contact_step is None
        ):
            self.first_two_finger_contact_step = step
            self.first_two_finger_contact_segment = segment
            first_two = True
        return first_any, first_two

    def as_dict(self) -> dict[str, int | None]:
        """Return stable JSON field names for structured run reporting."""
        return {
            "first_finger_contact_step": self.first_finger_contact_step,
            "first_finger_contact_segment": self.first_finger_contact_segment,
            "first_two_finger_contact_step": self.first_two_finger_contact_step,
            "first_two_finger_contact_segment": (
                self.first_two_finger_contact_segment
            ),
        }
