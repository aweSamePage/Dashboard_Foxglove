"""Foxglove debug publishers for the three student controllers.

Each class is used the same way: construct it on the ROS node, then call
``publish(...)`` every control step. Foxglove layouts subscribe to:

  /debug/mppi/*   MppiDebugPublisher    layout_mppi_debug.json
  /debug/mpc/*    MpcDebugPublisher     layout_mpc_debug.json
  /debug/ftg/*    FtgDebugPublisher     layout_ftg_debug.json

Advice strings go to ``.../advice``. Health tables go to ``.../health``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA, Float32, String
from visualization_msgs.msg import Marker, MarkerArray


# ---------------------------------------------------------------------------
# MPPI helper: sampling-health metric used by MppiDebugPublisher
# ---------------------------------------------------------------------------
def ess_ratio_from_rewards(
    total_reward: np.ndarray,
    temperature: float,
    damping: float,
    range_normalize: bool,
) -> float:
    """Effective sample size ratio (ESS / n_samples) of MPPI-style softmax weights.

    Replicates the solver weighting: range-normalize returns before softmax so the
    metric stays scale-invariant. Returns -1.0 when undefined.
    """
    n = int(total_reward.shape[0])
    if n == 0:
        return -1.0
    if temperature <= 0.0:
        temperature = 1.0
    r_max = float(np.max(total_reward))
    r_min = float(np.min(total_reward))
    denom = (r_max - r_min) + damping if range_normalize else 1.0
    if denom <= 0.0:
        denom = 1.0
    w = np.exp((total_reward - r_max) / denom / temperature)
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return -1.0
    w = w / w_sum
    ess = 1.0 / float(np.sum(w * w))
    return ess / float(n)


# ---------------------------------------------------------------------------
# MPPI — /debug/mppi/*  (Foxglove: layout_mppi_debug.json)
# Student node: construct once, then publish() after each plan().
# ---------------------------------------------------------------------------
class MppiDebugPublisher:
    """Algorithm-agnostic publisher for MPPI-style debug visualization.

    Feed it generic numpy arrays from any sampling-based controller; it publishes a
    standard ``/debug/mppi/*`` topic contract that the shared Foxglove layout consumes.
    Nothing here depends on a specific MPPI implementation.
    """

    def __init__(
        self,
        node,
        *,
        frame_id: str = "map",
        topic_prefix: str = "/debug/mppi",
        max_samples: int = 40,
        saturation_margin: float = 0.98,
        qos: int = 5,
    ) -> None:
        self._node = node
        self._frame_id = frame_id
        self._max_samples = max(1, int(max_samples))
        self._saturation_margin = float(saturation_margin)

        # Topics the MPPI Foxglove layout reads.
        self._samples_pub = node.create_publisher(MarkerArray, f"{topic_prefix}/samples", qos)
        self._chosen_pub = node.create_publisher(Path, f"{topic_prefix}/chosen", qos)
        self._health_pub = node.create_publisher(DiagnosticStatus, f"{topic_prefix}/health", qos)
        self._ess_pub = node.create_publisher(Float32, f"{topic_prefix}/ess_ratio", qos)
        self._cost_pub = node.create_publisher(Float32, f"{topic_prefix}/cost_mean", qos)
        self._steer_sat_pub = node.create_publisher(Float32, f"{topic_prefix}/steer_sat", qos)
        self._accel_sat_pub = node.create_publisher(Float32, f"{topic_prefix}/accel_sat", qos)
        self._advice_pub = node.create_publisher(String, f"{topic_prefix}/advice", qos)
        self._prev_cost_mean: float = 0.0
        self._warning_log: list = []
        self._last_tip_set: set = set()

    def publish(
        self,
        *,
        sampled_xy: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        chosen_xy: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        damping: float = 0.0,
        range_normalize: bool = True,
        steer: Optional[float] = None,
        steer_limit: Optional[float] = None,
        accel: Optional[float] = None,
        accel_limit: Optional[float] = None,
        speed: Optional[float] = None,
        speed_limit: Optional[float] = None,
    ) -> None:
        """Publish the standard debug topics.

        Args:
            sampled_xy: sampled rollout positions, shape [n_samples, horizon, >=2].
            rewards: per-sample reward, shape [n_samples, horizon] or [n_samples].
            chosen_xy: selected/optimal trajectory positions, shape [K, >=2].
            temperature, damping: MPPI weighting params (for the ESS metric).
            steer/accel/speed (+ *_limit): optional, used for saturation flags.
        """
        stamp = self._node.get_clock().now().to_msg()

        # 3D: grey sample rollouts + blue chosen path
        if sampled_xy is not None:
            arr = np.asarray(sampled_xy)
            if arr.ndim == 3 and arr.shape[0] > 0 and arr.shape[2] >= 2:
                self._samples_pub.publish(self._build_samples(arr, stamp))

        if chosen_xy is not None:
            ch = np.asarray(chosen_xy)
            if ch.ndim == 2 and ch.shape[0] > 0 and ch.shape[1] >= 2:
                self._chosen_pub.publish(self._build_chosen(ch, stamp))

        ess_ratio = -1.0
        cost_min = cost_mean = cost_max = 0.0
        if rewards is not None:
            r = np.asarray(rewards)
            total_reward = r.sum(axis=1) if r.ndim == 2 else (r if r.ndim == 1 else None)
            if total_reward is not None and total_reward.shape[0] > 0:
                cost = -total_reward
                cost_min = float(np.min(cost))
                cost_mean = float(np.mean(cost))
                cost_max = float(np.max(cost))
                ess_ratio = ess_ratio_from_rewards(
                    total_reward, float(temperature), float(damping), bool(range_normalize)
                )

        # Saturation flags for the health table / advice.
        steer_sat = self._saturated(steer, steer_limit)
        accel_sat = self._saturated(accel, accel_limit)
        speed_sat = self._saturated(speed, speed_limit, signed=False)

        status = DiagnosticStatus()
        status.name = "roboracer_dashboard_foxglove/mppi_health"
        status.hardware_id = "mppi"
        status.level = (
            DiagnosticStatus.WARN if (0.0 <= ess_ratio < 0.1) else DiagnosticStatus.OK
        )
        status.message = "mppi health"
        status.values = [
            KeyValue(key="ess_ratio", value=f"{ess_ratio:.4f}"),
            KeyValue(key="cost_min", value=f"{cost_min:.3f}"),
            KeyValue(key="cost_mean", value=f"{cost_mean:.3f}"),
            KeyValue(key="cost_max", value=f"{cost_max:.3f}"),
            KeyValue(key="steer_saturated", value=str(bool(steer_sat))),
            KeyValue(key="accel_saturated", value=str(bool(accel_sat))),
            KeyValue(key="speed_saturated", value=str(bool(speed_sat))),
        ]
        self._health_pub.publish(status)
        self._ess_pub.publish(Float32(data=float(ess_ratio)))
        self._cost_pub.publish(Float32(data=float(cost_mean)))
        self._steer_sat_pub.publish(Float32(data=1.0 if steer_sat else 0.0))
        self._accel_sat_pub.publish(Float32(data=1.0 if accel_sat else 0.0))
        self._advice_pub.publish(String(data=self._get_advice(
            ess_ratio, steer_sat, accel_sat, speed_sat, cost_mean
        )))

    # Advice rules (ESS collapse, saturation, cost spike). Appends new tips only.
    def _get_advice(
        self,
        ess_ratio: float,
        steer_sat: bool,
        accel_sat: bool,
        speed_sat: bool,
        cost_mean: float,
    ) -> str:
        import time as _time
        tips = []
        ess_low = 0.0 <= ess_ratio < 0.1

        if ess_low:
            tips.append("[ESS low] Raise the temperature parameter")
        if steer_sat and ess_low:
            tips.append("[Steer saturated] Fix ESS first (raise temperature)")
        if steer_sat and not ess_low:
            tips.append("[Steer saturated] Check yaw cost or corner waypoints; last resort: reduce the target/reference speed")
        if accel_sat or speed_sat:
            tips.append("[Speed saturated] Reduce the target/reference speed")
        if cost_mean > 0.0 and self._prev_cost_mean > 0.0:
            if cost_mean > 5.0 * self._prev_cost_mean:
                tips.append("[Cost spike] Car off line — check corner waypoints")
        self._prev_cost_mean = cost_mean if cost_mean > 0.0 else self._prev_cost_mean

        current_tips = set(tips)
        new_tips = current_tips - self._last_tip_set
        self._last_tip_set = current_tips
        for tip in tips:
            if tip in new_tips:
                ts = _time.strftime("%H:%M:%S")
                self._warning_log.append(f"[{ts}] {tip}")

        return "\n".join(self._warning_log) if self._warning_log else "OK"

    def _saturated(
        self, value: Optional[float], limit: Optional[float], signed: bool = True
    ) -> bool:
        if value is None or limit is None:
            return False
        limit = abs(float(limit))
        if limit <= 0.0:
            return False
        v = abs(float(value)) if signed else float(value)
        return v >= self._saturation_margin * limit

    def _build_samples(self, sampled_xy: np.ndarray, stamp) -> MarkerArray:
        n = sampled_xy.shape[0]
        step = max(1, n // self._max_samples)

        marker = Marker()
        marker.header.frame_id = self._frame_id
        marker.header.stamp = stamp
        marker.ns = "mppi_samples"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.35)
        marker.pose.orientation.w = 1.0

        for i in range(0, n, step):
            traj = sampled_xy[i]
            for t in range(traj.shape[0] - 1):
                marker.points.append(Point(x=float(traj[t, 0]), y=float(traj[t, 1]), z=0.0))
                marker.points.append(
                    Point(x=float(traj[t + 1, 0]), y=float(traj[t + 1, 1]), z=0.0)
                )

        return MarkerArray(markers=[marker])

    def _build_chosen(self, chosen_xy: np.ndarray, stamp) -> Path:
        path = Path()
        path.header.frame_id = self._frame_id
        path.header.stamp = stamp
        for k in range(chosen_xy.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = self._frame_id
            pose.header.stamp = stamp
            pose.pose.position.x = float(chosen_xy[k, 0])
            pose.pose.position.y = float(chosen_xy[k, 1])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path


# ---------------------------------------------------------------------------
# MPC — /debug/mpc/*  (Foxglove: layout_mpc_debug.json)
# Student node: construct once, then publish() after each control step.
# ---------------------------------------------------------------------------
class MpcDebugPublisher:
    """Generic debug publisher for any MPC-style controller.

    Publishes standard /debug/mpc/* topics consumed by layout_mpc_debug.json.
    The student passes steer_ratio, waypoint_dist, cost computed in their node.
    """

    def __init__(
        self,
        node,
        *,
        topic_prefix: str = "/debug/mpc",
        saturation_margin: float = 0.98,
        qos: int = 5,
    ) -> None:
        self._node = node
        self._saturation_margin = float(saturation_margin)
        self._health_pub = node.create_publisher(DiagnosticStatus, f"{topic_prefix}/health", qos)
        self._steer_ratio_pub = node.create_publisher(Float32, f"{topic_prefix}/steer_ratio", qos)
        self._dist_pub = node.create_publisher(Float32, f"{topic_prefix}/waypoint_dist", qos)
        self._cost_pub = node.create_publisher(Float32, f"{topic_prefix}/cost", qos)
        self._advice_pub = node.create_publisher(String, f"{topic_prefix}/advice", qos)
        self._prev_cost: float = 0.0
        self._prev_waypoint_dist: float = 0.0
        self._warning_log: list = []
        self._last_tip_set: set = set()

    def publish(
        self,
        *,
        steer_ratio: float = 0.0,
        waypoint_dist: float = 0.0,
        cost: float = 0.0,
        reacquire_dist: float = 1.0,
        steer_saturated: bool = False,
    ) -> None:
        self._steer_ratio_pub.publish(Float32(data=float(steer_ratio)))
        self._dist_pub.publish(Float32(data=float(waypoint_dist)))
        self._cost_pub.publish(Float32(data=float(cost)))

        off_track = waypoint_dist > reacquire_dist

        status = DiagnosticStatus()
        status.name = "roboracer_dashboard_foxglove/mpc_health"
        status.hardware_id = "simple_mpc"
        status.level = DiagnosticStatus.WARN if (off_track or steer_saturated) else DiagnosticStatus.OK
        status.message = "mpc health"
        status.values = [
            KeyValue(key="steer_ratio", value=f"{steer_ratio:.3f}"),
            KeyValue(key="waypoint_dist_m", value=f"{waypoint_dist:.3f}"),
            KeyValue(key="cost", value=f"{cost:.3f}"),
            KeyValue(key="steer_saturated", value=str(bool(steer_saturated))),
            KeyValue(key="off_track", value=str(bool(off_track))),
        ]
        self._health_pub.publish(status)
        self._advice_pub.publish(String(data=self._get_advice(
            steer_ratio, waypoint_dist, cost, reacquire_dist, steer_saturated
        )))

    # Advice rules (off-track, corner rollout, steer limit, cost spike).
    def _get_advice(
        self,
        steer_ratio: float,
        waypoint_dist: float,
        cost: float,
        reacquire_dist: float,
        steer_saturated: bool,
    ) -> str:
        import time as _time
        tips = []

        if waypoint_dist > reacquire_dist:
            tips.append("[Off-track] Car lost path — check corner waypoints or reduce target speed")
        dist_spike = waypoint_dist > 1.5 * max(self._prev_waypoint_dist, 0.1)
        self._prev_waypoint_dist = waypoint_dist if waypoint_dist > 0 else self._prev_waypoint_dist
        #if steer_saturated and dist_spike:
        if steer_saturated and waypoint_dist > 0.3:
            tips.append("[Corner rollout issue] Steer maxed when waypoint_dist jumped — "
                        "rollout is not anticipating the corner. "
                        "Pre-compute reference so it advances proportionally to speed each rollout step (no fixed index).")
        elif steer_saturated and cost > 80.0:
            tips.append("[High cost + steer maxed] Rollout may not see track correctly. "
                        "Try changing the waypoint spacing or interpolate the waypoints")
        elif steer_saturated:
            tips.append("[Steer saturated] Reduce cross-track-error cost weight or target speed in controller params")
        if steer_ratio > 0.85 and not steer_saturated:
            tips.append("[Steer near limit] Consider reducing cross-track-error cost weight or increasing max steering angle limit")
        if cost > 0.0 and self._prev_cost > 0.0 and cost > 5.0 * self._prev_cost:
            tips.append("[Cost spike] Rollout cost jumped — check waypoints near corners")
        self._prev_cost = cost if cost > 0.0 else self._prev_cost

        current_tips = set(tips)
        new_tips = current_tips - self._last_tip_set
        self._last_tip_set = current_tips
        for tip in tips:
            if tip in new_tips:
                ts = _time.strftime("%H:%M:%S")
                self._warning_log.append(f"[{ts}] {tip}")

        return "\n".join(self._warning_log) if self._warning_log else "OK"


# ---------------------------------------------------------------------------
# Follow The Gap — /debug/ftg/*  (Foxglove: layout_ftg_debug.json)
# 3D markers: green gap wedge, yellow AIM ball, red BUBBLE disc.
# ---------------------------------------------------------------------------
class FtgDebugPublisher:
    """Debug publisher for Follow-The-Gap.

    Foxglove 3D shows the gap (green), the aim point (yellow), and the
    safety bubble (red). Advice names those colors when something is wrong.
    """

    def __init__(
        self,
        node,
        *,
        topic_prefix: str = "/debug/ftg",
        collision_dist: float = 0.35,
        steer_jump_rad: float = 0.25, 
        persist_frames: int = 3,
        rearm_frames: int = 20, #was 10
        qos: int = 5,
    ) -> None:
        self._node = node
        self._collision_dist = float(collision_dist)
        self._steer_jump_rad = float(steer_jump_rad)
        self._persist_frames = max(1, int(persist_frames))
        self._rearm_frames = max(1, int(rearm_frames))
        self._steer_limit = 0.4189
        self._bubble_small_beams = 16.0  # Heuristic since radius=10 (20 beams) is good.
        self._bubble_large_beams = 80.0
        self._aim_wall = 0.8
        self._turning = 0.5
        self._chunk_ratio_min = 1.6
        self._chunk_ratio_max = 16.0
        self._rear_scan_rad = 2.0
        self._steer_as_deg = 1.5
        self._steer_idle = 0.06
        self._offset_jump_min = 0.12

        self._nearest_pub = node.create_publisher(Float32, f"{topic_prefix}/nearest_dist", qos)
        self._steer_deg_pub = node.create_publisher(Float32, f"{topic_prefix}/steer_deg", qos)
        self._best_offset_pub = node.create_publisher(Float32, f"{topic_prefix}/best_offset", qos)
        self._bubble_pub = node.create_publisher(Float32, f"{topic_prefix}/bubble_beams", qos)
        self._markers_pub = node.create_publisher(MarkerArray, f"{topic_prefix}/markers", qos)
        self._advice_pub = node.create_publisher(String, f"{topic_prefix}/advice", qos)
        self._health_pub = node.create_publisher(DiagnosticStatus, f"{topic_prefix}/health", qos)

        self._prev_steer: Optional[float] = None
        self._prev_offset: Optional[float] = None
        self._close_from_turn = False
        self._last_looks_chunked = False
        self._counters: dict = {}
        self._warning_log: list = []
        self._latched_tags: set = set()
        self._clear_counts: dict = {}

    def publish(
        self,
        *,
        steer: float = 0.0,
        speed: float = 0.0,
        scan=None,
        ranges: Optional[np.ndarray] = None,
        window_start: Optional[int] = None,
        gap=None,
        best_point: Optional[int] = None,
        nearest_index: int = -1,
        bubble_start: Optional[int] = None,
        bubble_end: Optional[int] = None,
        nearest_dist: float = -1.0,
        gap_width: float = 0.0,
        gap_start: int = -1,
        bubble_beams: float = -1.0,
        angle_increment: Optional[float] = None,
        angle_min: Optional[float] = None,
        frame_id: str = "laser",
    ) -> None:
        """Publish advice plus the 3D gap / aim / bubble markers.

        Students pass ``scan``, ``ranges``, ``gap``, and the indices they
        already have. Width / scan angles / nearest distance / bubble
        (zeroed run in ``ranges``) are derived here.
        """
        if scan is not None:
            if angle_increment is None:
                angle_increment = float(scan.angle_increment)
            if angle_min is None:
                angle_min = float(scan.angle_min)
            frame_id = str(getattr(scan.header, "frame_id", None) or frame_id)

        if window_start is None:
            window_start = self._infer_window_start(scan, ranges)

        gap_start, gap_width, best_point = self._unpack_gap(
            gap, gap_width, gap_start, best_point
        )
        if bubble_start is None and bubble_end is None:
            bubble_start, bubble_end, inferred_nearest = self._infer_bubble(ranges)
            if nearest_index < 0:
                nearest_index = inferred_nearest
        elif nearest_index < 0 and bubble_start is not None and bubble_end is not None:
            nearest_index = (int(bubble_start) + int(bubble_end)) // 2
        if bubble_start is not None and bubble_end is not None and float(bubble_beams) < 0.0:
            bubble_beams = float(max(0, int(bubble_end) - int(bubble_start)))
        elif ranges is not None and float(bubble_beams) < 0.0:
            bubble_beams = 0.0
        if ranges is not None and nearest_dist < 0.0:
            arr = np.asarray(ranges, dtype=float)
            finite = arr[np.isfinite(arr) & (arr > 0.0)]
            if finite.size:
                nearest_dist = float(np.min(finite))

        no_gap = float(gap_width) <= 0.0
        best_offset = self._aim_in_gap(float(gap_width), int(gap_start), int(best_point))
        gap_center = float(gap_start) + 0.5 * (float(gap_width) - 1.0)
        steer_ratio = abs(float(steer)) / self._steer_limit
        steer_deg = math.degrees(float(steer))
        steer_jump = 0.0 if self._prev_steer is None else abs(float(steer) - self._prev_steer)
        self._prev_steer = float(steer)
        if no_gap:
            offset_jump = 0.0
            self._prev_offset = None
        else:
            offset_jump = (
                0.0
                if self._prev_offset is None
                else abs(float(best_offset) - self._prev_offset)
            )
            self._prev_offset = float(best_offset)
        expected_steer = 0.0
        looks_chunked = self._last_looks_chunked
        if ranges is not None and angle_increment is not None and int(best_point) >= 0:
            n = int(np.asarray(ranges).shape[0])
            if n > 0:
                expected_steer = (int(best_point) - n // 2) * float(angle_increment)
                looks_chunked = self._looks_chunked(
                    self._chunk_ratio(n, int(best_point), float(steer), float(angle_increment))
                )
                self._last_looks_chunked = looks_chunked
        looks_rear = self._looks_rear(
            ranges, angle_min, angle_increment, int(window_start)
        )

        self._nearest_pub.publish(Float32(data=float(nearest_dist)))
        self._steer_deg_pub.publish(Float32(data=float(steer_deg)))
        self._best_offset_pub.publish(Float32(data=float(best_offset)))
        self._bubble_pub.publish(Float32(data=float(bubble_beams)))

        if (
            ranges is not None
            and angle_increment is not None
            and angle_min is not None
            and hasattr(self._node, "get_clock")
        ):
            self._markers_pub.publish(self._build_markers(
                ranges=np.asarray(ranges, dtype=float),
                angle_increment=float(angle_increment),
                angle_min=float(angle_min),
                window_start=int(window_start),
                frame_id=str(frame_id),
                gap_start=int(gap_start),
                gap_width=float(gap_width),
                best_point=int(best_point),
                nearest_index=int(nearest_index),
                nearest_dist=float(nearest_dist),
                bubble_start=bubble_start,
                bubble_end=bubble_end,
                bubble_beams=float(bubble_beams),
            ))

        too_close = 0.0 <= nearest_dist < self._collision_dist
        status = DiagnosticStatus()
        status.name = "roboracer_dashboard_foxglove/ftg_health"
        status.hardware_id = "follow_the_gap"
        status.level = DiagnosticStatus.WARN if (no_gap or too_close) else DiagnosticStatus.OK
        status.message = "ftg health"
        status.values = [
            KeyValue(key="nearest_dist_m", value=f"{nearest_dist:.3f}"),
            KeyValue(key="steer_deg", value=f"{steer_deg:+.1f}"),
            KeyValue(key="best_offset", value=f"{best_offset:+.2f}"),
            KeyValue(key="bubble_beams", value=f"{float(bubble_beams):.0f}"),
        ]
        self._health_pub.publish(status)

        self._advice_pub.publish(String(data=self._get_advice(
            no_gap=no_gap,
            nearest_dist=float(nearest_dist),
            bubble_beams=float(bubble_beams),
            best_offset=best_offset,
            steer_ratio=steer_ratio,
            steer_jump=steer_jump,
            offset_jump=offset_jump,
            speed=float(speed),
            steer=float(steer),
            expected_steer=expected_steer,
            looks_chunked=looks_chunked,
            looks_rear=looks_rear,
            best_point=int(best_point),
            gap_center=gap_center,
        )))

    @staticmethod
    def _unpack_gap(gap, gap_width: float, gap_start: int, best_point: Optional[int]):
        """Turn ``gap=(start, end)`` or ``(None, None)`` into start / width / aim."""
        if gap is not None:
            if gap[0] is None:
                return -1, 0.0, -1 if best_point is None else int(best_point)
            start_i, end_i = int(gap[0]), int(gap[1])
            width = float(end_i - start_i + 1)
            if best_point is None or int(best_point) < 0:
                aim = int(round(0.5 * (start_i + end_i)))
            else:
                aim = int(best_point)
            return start_i, width, aim
        aim = -1 if best_point is None else int(best_point)
        return int(gap_start), float(gap_width), aim

    @staticmethod
    def _chunk_ratio(n_ranges: int, best_point: int, steer: float, angle_increment: float) -> float:
        """len(ranges) / processed length implied by the lab steer formula."""
        if n_ranges <= 0 or abs(angle_increment) < 1e-9:
            return 1.0
        n_proc = 2.0 * (float(best_point) - float(steer) / float(angle_increment))
        if n_proc < 8.0:
            return 1.0
        return float(n_ranges) / n_proc

    def _looks_chunked(self, chunk_ratio: float) -> bool:
        return self._chunk_ratio_min <= float(chunk_ratio) <= self._chunk_ratio_max

    @staticmethod
    def _infer_window_start(scan, ranges) -> int:
        """Centered crop: processed index 0 is the first beam of that window."""
        if scan is None or ranges is None:
            return 0
        scan_ranges = getattr(scan, "ranges", None)
        if scan_ranges is None:
            return 0
        n_scan = len(scan_ranges)
        n = int(np.asarray(ranges).shape[0])
        if n_scan <= 0 or n <= 0 or n > n_scan:
            return 0
        return (n_scan - n) // 2

    @staticmethod
    def _infer_bubble(ranges):
        """Longest <=0 run in published ranges. Slice is [start, end)."""
        if ranges is None:
            return None, None, -1
        arr = np.asarray(ranges, dtype=float)
        n = int(arr.shape[0])
        if n <= 0:
            return None, None, -1
        cleared = ~np.isfinite(arr) | (arr <= 0.0)
        best_s = best_e = -1
        best_n = 0
        i = 0
        while i < n:
            if not cleared[i]:
                i += 1
                continue
            j = i + 1
            while j < n and cleared[j]:
                j += 1
            if j - i > best_n:
                best_n = j - i
                best_s, best_e = i, j
            i = j
        if best_n <= 0:
            return None, None, -1
        return best_s, best_e, best_s + best_n // 2

    @staticmethod
    def _bubble_wall_range(ranges, bubble_start, bubble_end) -> Optional[float]:
        """Closest positive range just outside the zeroed bubble."""
        if ranges is None or bubble_start is None or bubble_end is None:
            return None
        arr = np.asarray(ranges, dtype=float)
        n = int(arr.shape[0])
        best = None
        for i in (int(bubble_start) - 1, int(bubble_end)):
            if 0 <= i < n:
                r = float(arr[i])
                if math.isfinite(r) and r > 0.05 and (best is None or r < best):
                    best = r
        return best

    def _looks_rear(self, ranges, angle_min, angle_increment, window_start: int) -> bool:
        if ranges is None or angle_min is None or angle_increment is None:
            return False
        n = int(np.asarray(ranges).shape[0])
        if n <= 0:
            return False
        start = float(angle_min) + float(window_start) * float(angle_increment)
        end = float(angle_min) + float(window_start + n - 1) * float(angle_increment)
        return abs(start) > self._rear_scan_rad or abs(end) > self._rear_scan_rad

    @staticmethod
    def _aim_in_gap(gap_width: float, gap_start: int, best_point: int) -> float:
        """-1 = left wall of the gap, 0 = midpoint, +1 = right wall."""
        if gap_width <= 0.0 or gap_start < 0 or best_point < 0:
            return 0.0
        gap_center = float(gap_start) + 0.5 * (gap_width - 1.0)
        half = max(0.5 * gap_width, 1e-6)
        return float(np.clip((float(best_point) - gap_center) / half, -1.0, 1.0))

    @staticmethod
    def _beam_xy(index: int, ranges: np.ndarray, angle_min: float, angle_inc: float, window_start: int):
        if index < 0 or index >= int(ranges.shape[0]):
            return None
        r = float(ranges[index])
        if not math.isfinite(r) or r <= 0.05:
            return None
        angle = angle_min + float(window_start + index) * angle_inc
        return (r * math.cos(angle), r * math.sin(angle))

    def _build_markers(
        self,
        *,
        ranges: np.ndarray,
        angle_increment: float,
        angle_min: float,
        window_start: int,
        frame_id: str,
        gap_start: int,
        gap_width: float,
        best_point: int,
        nearest_index: int,
        nearest_dist: float,
        bubble_start: Optional[int],
        bubble_end: Optional[int],
        bubble_beams: float,
    ) -> MarkerArray:
        stamp = self._node.get_clock().now().to_msg()
        gap_end = int(gap_start + gap_width - 1) if gap_width > 0.0 and gap_start >= 0 else -1
        gap_ok = gap_end >= gap_start >= 0
        origin = Point(x=0.0, y=0.0, z=0.04)

        def xy(i: int):
            return self._beam_xy(i, ranges, angle_min, angle_increment, window_start)

        def base(mid: int, mtype: int) -> Marker:
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns = "ftg"
            m.id = mid
            m.type = mtype
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            return m

        # Green pizza slice = the free gap the car chose.
        fan = base(0, Marker.TRIANGLE_LIST)
        fan.scale.x = fan.scale.y = fan.scale.z = 1.0
        fan.color = ColorRGBA(r=0.16, g=0.83, b=0.40, a=0.30)
        if gap_ok:
            step = max(1, (gap_end - gap_start) // 20)
            pts = [xy(i) for i in range(gap_start, gap_end + 1, step)]
            last = xy(gap_end)
            if last is not None and (not pts or pts[-1] != last):
                pts.append(last)
            pts = [p for p in pts if p is not None]
            for a, b in zip(pts, pts[1:]):
                fan.points.extend((
                    origin,
                    Point(x=a[0], y=a[1], z=0.04),
                    Point(x=b[0], y=b[1], z=0.04),
                ))
        if not fan.points:
            fan.action = Marker.DELETE

        # Yellow AIM ball + beam — flickers on wobble, sits on a wall in a bad corner.
        aim_xy = xy(best_point) if best_point >= 0 else None
        ball = base(1, Marker.SPHERE)
        beam = base(2, Marker.LINE_LIST)
        label = base(3, Marker.TEXT_VIEW_FACING)
        if aim_xy is not None:
            ball.pose.position.x, ball.pose.position.y = aim_xy
            ball.pose.position.z = 0.18
            ball.scale.x = ball.scale.y = ball.scale.z = 0.28
            ball.color = ColorRGBA(r=1.0, g=0.84, b=0.15, a=0.95)
            beam.scale.x = 0.06
            beam.color = ColorRGBA(r=1.0, g=0.84, b=0.15, a=0.90)
            beam.points.extend((origin, Point(x=aim_xy[0], y=aim_xy[1], z=0.10)))
            label.pose.position.x, label.pose.position.y = aim_xy
            label.pose.position.z = 0.42
            label.scale.z = 0.22
            label.color = ColorRGBA(r=1.0, g=0.92, b=0.40, a=1.0)
            label.text = "AIM"
        else:
            ball.action = beam.action = label.action = Marker.DELETE

        # Red disc around the closest obstacle = the safety bubble.
        near_xy = xy(nearest_index)
        wall_r = self._bubble_wall_range(ranges, bubble_start, bubble_end)
        if near_xy is None and 0 <= nearest_index < int(ranges.shape[0]):
            r = wall_r if wall_r is not None else (
                nearest_dist if nearest_dist > 0.05 else 0.2
            )
            angle = angle_min + float(window_start + nearest_index) * angle_increment
            near_xy = (r * math.cos(angle), r * math.sin(angle))
        disc = base(4, Marker.CYLINDER)
        bubble_label = base(5, Marker.TEXT_VIEW_FACING)
        if near_xy is not None:
            half_beams = 0.0
            if bubble_start is not None and bubble_end is not None:
                half_beams = 0.5 * max(0, int(bubble_end) - int(bubble_start))
            elif bubble_beams > 0.0:
                half_beams = 0.5 * bubble_beams
            dist = math.hypot(*near_xy)
            # 6x for Foxglove visibility; not the true meter width of the beams.
            radius = max(0.05, 6 * dist * math.tan(max(half_beams, 1.0) * angle_increment))
            disc.pose.position.x, disc.pose.position.y = near_xy
            disc.pose.position.z = 0.03
            disc.scale.x = disc.scale.y = 2.0 * radius
            disc.scale.z = 0.07
            disc.color = ColorRGBA(r=0.96, g=0.25, b=0.25, a=0.45)
            bubble_label.pose.position.x, bubble_label.pose.position.y = near_xy
            bubble_label.pose.position.z = 0.28
            bubble_label.scale.z = 0.18
            bubble_label.color = ColorRGBA(r=1.0, g=0.55, b=0.55, a=1.0)
            bubble_label.text = "BUBBLE"
        else:
            disc.action = bubble_label.action = Marker.DELETE

        return MarkerArray(markers=[fan, ball, beam, label, disc, bubble_label])

    def _persisted(self, key: str, condition: bool) -> bool:
        """True once ``condition`` held for ``persist_frames`` consecutive calls."""
        count = self._counters.get(key, 0) + 1 if condition else 0
        self._counters[key] = count
        return count >= self._persist_frames

    def _get_advice(
        self,
        *,
        no_gap: bool,
        nearest_dist: float,
        bubble_beams: float,
        best_offset: float,
        steer_ratio: float,
        steer_jump: float,
        offset_jump: float = 0.0,
        speed: float,
        steer: float = 0.0,
        expected_steer: float = 0.0,
        looks_chunked: bool = False,
        looks_rear: bool = False,
        best_point: int = -1,
        gap_center: float = 0.0,
    ) -> str:
        import time as _time
        tips = []
        too_close = 0.0 <= nearest_dist < self._collision_dist
        looks_degrees = abs(float(steer)) > self._steer_as_deg
        if abs(expected_steer) > 0.15 and abs(float(steer) - expected_steer) <= 0.15:
            looks_degrees = False
        turning = (not looks_degrees) and steer_ratio > self._turning
        bubble_small = 0.0 <= bubble_beams < self._bubble_small_beams
        bubble_large = bubble_beams >= self._bubble_large_beams
        aiming_wall = abs(best_offset) > self._aim_wall
        if turning:
            if too_close:
                self._close_from_turn = True
        elif not too_close:
            self._close_from_turn = False

        chunk_tip = (
            "[Chunking] The green gap and yellow AIM do not match the "
            "array you passed as ranges. If you preprocess or chunk, "
            "pass that processed lidar array, not the raw slice."
        )
        chunking = self._persisted("chunking", bool(looks_chunked))
        if chunking:
            tips.append(chunk_tip)

        if self._persisted("rear_scan", (not chunking) and bool(looks_rear)):
            tips.append(
                "[Rear scan] The lidar window includes beams behind the car. "
                "Use only the forward slice."
            )

        if self._persisted("steer_units", (not chunking) and looks_degrees):
            tips.append(
                "[Steer units] Steering is far larger than a typical car "
                "angle. Use radians, not degrees or a beam index."
            )

        if self._persisted(
            "steer_unused",
            (not chunking)
            and (not looks_degrees)
            and abs(float(steer)) < self._steer_idle
            and abs(expected_steer) > 0.15,
        ):
            tips.append(
                "[Steer unused] The yellow AIM is off to the side but "
                "steering is almost zero. Convert the AIM beam to a "
                "steering angle in radians."
            )

        # Sign flipped: AIM left (+) but steer is right, or the reverse.
        if self._persisted(
            "steer_sign",
            (not chunking)
            and abs(expected_steer) > 0.15
            and steer * expected_steer < 0,
        ):
            tips.append(
                "[Steer sign] Steering is the opposite of the yellow AIM. "
                "Left is positive."
            )

        jumping = float(offset_jump) > self._offset_jump_min
        # Straight wobble: yellow AIM ball jumps left/right in a corridor.
        if self._persisted(
            "straight_wobble",
            (not chunking) and not turning and jumping,
        ):
            tips.append(
                "[Straight wobble] The yellow AIM ball is jumping left/right. "
                "Aim at the "
                "middle of the green gap"
            )

        if self._persisted(
            "far_aim",
            (not chunking)
            and not turning
            and (not jumping)
            and abs(best_offset) > 0.4
            and abs(best_point - gap_center) > 25,
        ):
            tips.append(
                "[Far AIM] The yellow AIM is not in the middle of the green "
                "gap. Aim at the gap midpoint, not the farthest beam."
            )

        # Bubble ate the scan: leftover gap is gone or AIM sits on a wall.
        if self._persisted("bubble_large", bubble_large and (no_gap or aiming_wall)):
            tips.append(
                "[Bubble too large] The red BUBBLE ate the gap. "
                "Shrink the safety bubble around the closest obstacle."
            )

        # Bubble too small: new scrape on a straight, not leftover from a corner.
        if self._persisted(
            "bubble_small",
            too_close and not turning and bubble_small and not self._close_from_turn,
        ):
            tips.append(
                "[Bubble too small] The red BUBBLE is tiny. "
                "Increase the safety bubble around the closest obstacle."
            )

        # Corner: AIM on the gap edge, and/or still fast while steering hard.
        if self._persisted("corner_aim", (not chunking) and turning and aiming_wall):
            tips.append(
                "[Corner AIM] The yellow AIM ball is on the wall (edge of "
                "the green gap). Use the gap midpoint."
            )
        if self._persisted(
            "corner_speed",
            (not chunking) and turning and speed > 3.0 and too_close,
        ):
            tips.append(
                "[Corner speed] Steering is large and speed is still high. "
                "Scale speed down when the steering angle is large."
            )

        # Rare fallback: find_max_gap returned nothing (not a huge bubble).
        if self._persisted("no_gap", no_gap and not bubble_large):
            tips.append(
                "[No gap] No beam passed the free-space threshold. "
                "Lower the free-space distance cutoff, or raise the "
                "max lidar range clip."
            )

        current_tags = {self._advice_tag(tip) for tip in tips}
        for tag in list(self._latched_tags):
            if tag in current_tags:
                self._clear_counts[tag] = 0
            else:
                self._clear_counts[tag] = self._clear_counts.get(tag, 0) + 1
                if self._clear_counts[tag] >= self._rearm_frames:
                    self._latched_tags.discard(tag)
                    self._clear_counts.pop(tag, None)

        new_other = False
        for tip in tips:
            tag = self._advice_tag(tip)
            if tag == "Chunking":
                continue
            if tag not in self._latched_tags:
                self._latched_tags.add(tag)
                self._clear_counts[tag] = 0
                ts = _time.strftime("%H:%M:%S")
                self._warning_log.append(f"[{ts}] {tip}")
                new_other = True

        last_is_chunking = bool(
            self._warning_log and "[Chunking]" in self._warning_log[-1]
        )
        if chunking and (not last_is_chunking) and (
            "Chunking" not in self._latched_tags or new_other
        ):
            self._latched_tags.add("Chunking")
            self._clear_counts["Chunking"] = 0
            ts = _time.strftime("%H:%M:%S")
            self._warning_log.append(f"[{ts}] {chunk_tip}")
        del self._warning_log[:-100]

        return "\n".join(self._warning_log) if self._warning_log else "OK"

    @staticmethod
    def _advice_tag(tip: str) -> str:
        if tip.startswith("[") and "]" in tip:
            return tip[1:tip.index("]")]
        return tip


# ---------------------------------------------------------------------------
# Adapter: f1tenth_planning Dynamic_MPPI_Planner -> MppiDebugPublisher.publish()
# ---------------------------------------------------------------------------
def extract_from_f1tenth_planning(planner) -> dict:
    """Adapter: pull generic debug arrays out of a f1tenth_planning MPPI planner.

    Returns a dict ready to splat into ``MppiDebugPublisher.publish(**...)``. Reads
    attributes defensively so a planner-version change never breaks the control path.
    Works for any planner exposing ``solver.samples`` / ``x_pred`` in the same layout.
    """
    out = {
        "sampled_xy": None,
        "rewards": None,
        "chosen_xy": None,
        "temperature": 1.0,
        "damping": 0.0,
    }

    solver = getattr(planner, "solver", None)
    samples = getattr(solver, "samples", None) if solver is not None else None
    if samples is not None and len(samples) >= 3:
        s_sampled = np.asarray(samples[1])  # [n_samples, N, nx]
        if s_sampled.ndim == 3 and s_sampled.shape[2] >= 2:
            out["sampled_xy"] = s_sampled[:, :, :2]
        out["rewards"] = np.asarray(samples[2])  # [n_samples, N]

    x_pred = getattr(planner, "x_pred", None)
    if x_pred is not None:
        x_arr = np.asarray(x_pred)  # [nx, N+1]
        if x_arr.ndim == 2 and x_arr.shape[0] >= 2:
            out["chosen_xy"] = x_arr[:2, :].T  # [N+1, 2]

    config = getattr(solver, "config", None)
    out["temperature"] = float(getattr(config, "temperature", 1.0))
    out["damping"] = float(getattr(config, "damping", 0.0))
    return out
