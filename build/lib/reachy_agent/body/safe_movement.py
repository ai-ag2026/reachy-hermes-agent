"""Safe bounded head movement for AGENT-controlled Reachy (M3 policy + P0 corrected execution).

The v0.1 safety policy is unchanged (see `movement_policy.plan_agent_movement`): a curated allowlist
(look_left/right/up/down/front, stop_motion), bounded relative head deltas (±5° yaw / ±4° pitch), and
**antennas + body_yaw always preserved**. What changed (2026-06-24 audits, docs/audits/2026-06-24/):

- Execution is now abstracted behind a small ``Mover`` protocol so the SAME policy drives either:
  * the official app's **MovementManager** — the LIVE runtime path: a 60 Hz worker loop whose single
    control point is ``ReachyMini.set_target``. Bounded moves are ENQUEUED via ``queue_move`` +
    ``set_moving_state`` (``ManagerMover``); the app's ``GotoQueueMove`` is injected as a factory so
    this package stays decoupled from the app. THIS is how movement should run on the robot.
  * the SDK directly via ``goto_target`` (``SdkMover``) — standalone/diagnostic only. NOTE: requires
    ``enable_motors()`` BEFORE any target (the SDK pins targets to the present pose on enable), and a
    single ``goto_target`` only holds for its duration; it is NOT the live "alive" path.
- ``look_front`` is an **absolute recenter** to the neutral head pose (identity), not a relative
  zero-delta (which was a no-op that never recentered). Body_yaw + antennas stay preserved.

``create_head_pose`` is imported lazily (and is injectable) so this module is unit-testable without the
``reachy_mini`` package installed.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import numpy as np

from .movement_policy import plan_agent_movement

logger = logging.getLogger(__name__)

# (x, y, z, roll, pitch, yaw) in degrees — bounded curated deltas. `front` is handled as an
# absolute recenter (not a relative delta), so it is intentionally not in this table.
_SAFE_DELTAS: dict[str, tuple[int, int, int, int, int, int]] = {
    "left": (0, 0, 0, 0, 0, 5),
    "right": (0, 0, 0, 0, 0, -5),
    "up": (0, 0, 0, 0, -4, 0),
    "down": (0, 0, 0, 0, 4, 0),
}

HeadPoseFactory = Callable[..., np.ndarray]


def _default_create_head_pose(*args: Any, **kwargs: Any) -> np.ndarray:
    from reachy_mini.utils import create_head_pose  # lazy: only needed on robot/app venv

    return create_head_pose(*args, **kwargs)


def _first_float(value: Any) -> float:
    try:
        return float(value[0])
    except (TypeError, IndexError):
        return float(value)


class Mover(Protocol):
    """Execution seam for a bounded move. Adapters: ManagerMover (live) / SdkMover (standalone)."""

    def current_head_pose(self) -> np.ndarray: ...
    def current_body_yaw_antennas(self) -> Tuple[float, Tuple[float, float]]: ...
    def queue_goto(
        self,
        *,
        target_head_pose: np.ndarray,
        start_head_pose: np.ndarray,
        antennas: Tuple[float, float],
        body_yaw: float,
        duration: float,
    ) -> Any: ...
    def stop(self) -> Any: ...


class ManagerMover:
    """Enqueue bounded moves into the official app MovementManager (the live 60 Hz runtime path).

    ``goto_move_factory`` builds the app's ``GotoQueueMove`` (inject ``dance_emotion_moves.GotoQueueMove``)
    so this package never imports the app. Mirrors the app's ``tools/agent_safe_movement._queue_safe_head_motion``.
    """

    def __init__(self, reachy: Any, movement_manager: Any, goto_move_factory: Callable[..., Any]) -> None:
        self.reachy = reachy
        self.mm = movement_manager
        self._factory = goto_move_factory

    def current_head_pose(self) -> np.ndarray:
        return np.asarray(self.reachy.get_current_head_pose()).astype("float32")

    def current_body_yaw_antennas(self) -> Tuple[float, Tuple[float, float]]:
        body_yaw, antennas = self.reachy.get_current_joint_positions()
        return _first_float(body_yaw), (float(antennas[0]), float(antennas[1]))

    def queue_goto(
        self,
        *,
        target_head_pose: np.ndarray,
        start_head_pose: np.ndarray,
        antennas: Tuple[float, float],
        body_yaw: float,
        duration: float,
    ) -> Any:
        move = self._factory(
            target_head_pose=target_head_pose,
            start_head_pose=start_head_pose,
            target_antennas=antennas,
            start_antennas=antennas,
            target_body_yaw=body_yaw,
            start_body_yaw=body_yaw,
            duration=duration,
        )
        self.mm.queue_move(move)
        self.mm.set_moving_state(duration)
        return {"queued": True}

    def stop(self) -> Any:
        return self.mm.clear_move_queue()


class SdkMover:
    """Drive ReachyMini directly via goto_target. Standalone/diagnostic only — NOT the live path.

    Requires ``enable_motors()`` to have been called first (the SDK pins targets to the present pose
    on enable). A single goto only holds for its duration.
    """

    def __init__(self, reachy: Any) -> None:
        self.reachy = reachy

    def current_head_pose(self) -> np.ndarray:
        return np.asarray(self.reachy.get_current_head_pose())

    def current_body_yaw_antennas(self) -> Tuple[float, Tuple[float, float]]:
        body_yaw, antennas = self.reachy.get_current_joint_positions()
        return _first_float(body_yaw), (float(antennas[0]), float(antennas[1]))

    def queue_goto(
        self,
        *,
        target_head_pose: np.ndarray,
        start_head_pose: np.ndarray,
        antennas: Tuple[float, float],
        body_yaw: float,
        duration: float,
    ) -> Any:
        return self.reachy.goto_target(
            head=target_head_pose, antennas=list(antennas), body_yaw=body_yaw, duration=duration
        )

    def stop(self) -> Any:
        head = self.current_head_pose()
        body_yaw, antennas = self.current_body_yaw_antennas()
        return self.reachy.goto_target(head=head, antennas=list(antennas), body_yaw=body_yaw, duration=0.2)


def _as_mover(target: Any) -> Mover:
    """A real Mover is passed through; a raw ReachyMini is wrapped in SdkMover (back-compat)."""
    return target if hasattr(target, "queue_goto") else SdkMover(target)


def apply_safe_movement(
    target: Any,
    intent: str,
    *,
    duration: float = 0.3,
    reason: str = "",
    head_pose_factory: Optional[HeadPoseFactory] = None,
) -> Dict[str, Any]:
    """Plan + execute one curated, bounded movement. ``target`` is a Mover or a raw ReachyMini.

    Returns a metadata dict (status executed/blocked/error, intent, bounded_degrees, preserved,
    side_effects). Antennas + body_yaw are always preserved. ``look_front`` recenters the head to the
    absolute neutral pose; other looks compose a bounded relative delta onto the current pose.
    """
    chp = head_pose_factory or _default_create_head_pose
    plan = plan_agent_movement(intent, reason=reason)
    if plan.status != "planned":
        return {
            "status": "blocked",
            "intent": plan.intent,
            "reason": plan.reason or "intent not in curated v0.1 allowlist",
            "side_effects": [],
        }

    # Clamp duration to a sane window: 0/negative/NaN would make the downstream interpolation
    # snap (t/duration) or corrupt the manager's moving timer; too long = a crawling move.
    if not math.isfinite(duration):
        duration = 0.3
    duration = min(max(float(duration), 0.15), 2.0)

    mover = _as_mover(target)

    # Any mover call can raise on a daemon disconnect / zenoh timeout mid-move. The docstring
    # promises an "error" status, so translate a failure into one instead of letting it escape
    # into the tool layer (which would crash the voice turn). The stop path is guarded too.
    try:
        if plan.selected_tool == "clear_move_queue":  # stop_motion -> halt
            mover.stop()
            logger.info("agent safe_movement: stop_motion (halt)")
            return {
                "status": "executed",
                "intent": plan.intent,
                "side_effects": ["movement_stopped"],
                "preserved": ["antennas", "body_yaw"],
            }

        direction = plan.tool_args["direction"]
        body_yaw, antennas = mover.current_body_yaw_antennas()
        start = np.asarray(mover.current_head_pose())

        # A daemon that returns a non-finite pose (partial failure/reconnect) would propagate
        # NaN/Inf through the matmul into set_target — undefined motion despite "bounded" policy.
        if not np.all(np.isfinite(start)):
            logger.warning("agent safe_movement: non-finite current head pose -> abort")
            return {"status": "error", "intent": plan.intent, "error": "non-finite head pose", "side_effects": []}

        if direction == "front":  # absolute recenter to neutral (not a relative zero-delta no-op)
            target_pose = np.asarray(chp(0, 0, 0, 0, 0, 0, degrees=True))
            bounded = {"recenter": "absolute_neutral", "yaw": 0, "pitch": 0}
        else:
            dx, dy, dz, droll, dpitch, dyaw = _SAFE_DELTAS[direction]
            target_pose = np.matmul(start, np.asarray(chp(dx, dy, dz, droll, dpitch, dyaw, degrees=True)))
            bounded = {"x": dx, "y": dy, "z": dz, "roll": droll, "pitch": dpitch, "yaw": dyaw}

        if not np.all(np.isfinite(target_pose)):
            logger.warning("agent safe_movement: non-finite target pose -> abort")
            return {"status": "error", "intent": plan.intent, "error": "non-finite target pose", "side_effects": []}

        mover.queue_goto(
            target_head_pose=target_pose,
            start_head_pose=start,
            antennas=antennas,
            body_yaw=body_yaw,
            duration=duration,
        )
    except Exception as exc:
        logger.warning("agent safe_movement: mover call failed: %r", exc)
        return {"status": "error", "intent": plan.intent, "error": str(exc), "side_effects": []}

    logger.info("agent safe_movement: %s (antennas/body_yaw preserved)", plan.intent)
    return {
        "status": "executed",
        "intent": plan.intent,
        "direction": direction,
        "bounded_degrees": bounded,
        "duration_s": duration,
        "side_effects": ["head_moved"],
        "preserved": ["antennas", "body_yaw"],
    }
