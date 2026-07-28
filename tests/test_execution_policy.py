import unittest

from scripts.ik.execution_policy import (
    DescentContactDiagnostics,
    distributed_command_steps,
)


class ExecutionPolicyTests(unittest.TestCase):
    def test_command_steps_are_distributed_without_losing_budget(self) -> None:
        steps = distributed_command_steps(300, 8)
        self.assertEqual(steps, [38, 38, 38, 38, 37, 37, 37, 37])
        self.assertEqual(sum(steps), 300)

    def test_contact_diagnostics_record_each_first_event_once(self) -> None:
        diagnostics = DescentContactDiagnostics()
        self.assertEqual(
            diagnostics.observe(
                segment=5,
                step=3,
                fixed_contact=True,
                moving_contact=False,
            ),
            (True, False),
        )
        self.assertEqual(
            diagnostics.observe(
                segment=7,
                step=18,
                fixed_contact=True,
                moving_contact=True,
            ),
            (False, True),
        )
        self.assertEqual(
            diagnostics.observe(
                segment=8,
                step=1,
                fixed_contact=True,
                moving_contact=True,
            ),
            (False, False),
        )
        self.assertEqual(
            diagnostics.as_dict(),
            {
                "first_finger_contact_step": 3,
                "first_finger_contact_segment": 5,
                "first_two_finger_contact_step": 18,
                "first_two_finger_contact_segment": 7,
            },
        )


if __name__ == "__main__":
    unittest.main()
