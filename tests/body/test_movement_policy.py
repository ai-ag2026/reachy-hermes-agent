"""Pure unit tests for the curated movement allowlist (no SDK / no robot)."""

from reachy_agent.body.movement_policy import plan_agent_movement


def test_look_intents_are_planned_with_bounded_direction():
    for intent, direction in [
        ("look_left", "left"),
        ("look_right", "right"),
        ("look_up", "up"),
        ("look_down", "down"),
        ("look_front", "front"),
    ]:
        plan = plan_agent_movement(intent)
        assert plan.status == "planned"
        assert plan.selected_tool == "agent_safe_head_motion"
        assert plan.tool_args["direction"] == direction
        assert plan.requires_live_execute is True


def test_stop_motion_maps_to_halt_seam():
    plan = plan_agent_movement("stop_motion")
    assert plan.status == "planned"
    assert plan.selected_tool == "clear_move_queue"
    assert plan.side_effects == []


def test_intents_outside_allowlist_are_blocked_without_side_effects():
    for bad in ["dance", "spin", "look_around", "wave", "", "drop table", "look_left; rm -rf"]:
        plan = plan_agent_movement(bad)
        assert plan.status == "blocked", bad
        assert plan.selected_tool is None
        assert plan.requires_live_execute is False
        assert plan.side_effects == []


def test_bad_input_does_not_raise():
    for value in [None, "  ", "LOOK_LEFT  ", " Stop_Motion "]:
        plan = plan_agent_movement(value)  # type: ignore[arg-type]
        assert plan.status in {"planned", "blocked"}
    # case-insensitive + trimmed
    assert plan_agent_movement("  LOOK_LEFT ").status == "planned"
