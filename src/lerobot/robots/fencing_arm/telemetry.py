# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import csv
import os
import time
from pathlib import Path

from .poses import FENCING_ARM_JOINT_KEYS

MOTOR_NAMES = tuple(key.removesuffix(".pos") for key in FENCING_ARM_JOINT_KEYS)


class FencingArmTelemetryRecorder:
    """Incrementally persist servo telemetry and render it after a run."""

    def __init__(self, output_dir: Path, sample_fps: float = 5.0):
        if sample_fps <= 0:
            raise ValueError("telemetry sample_fps must be greater than zero")
        self.output_dir = output_dir
        self.sample_period = 1 / sample_fps
        self.csv_path = output_dir / "telemetry.csv"
        self.started_at = time.monotonic()
        self.next_sample_at = self.started_at
        self.last_sync_at = self.started_at
        output_dir.mkdir(parents=True, exist_ok=True)
        self.file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=(
                "time_s",
                "motor",
                "position_deg",
                "goal_position_deg",
                "position_error_deg",
                "velocity_raw",
                "voltage_v",
                "current_raw",
                "torque_load_percent",
                "temperature_c",
            ),
        )
        self.writer.writeheader()
        self.file.flush()

    def capture(self, bus, positions: dict[str, float]) -> None:
        now = time.monotonic()
        if now < self.next_sample_at:
            return
        self.next_sample_at = now + self.sample_period
        values = {
            "goal_position": bus.sync_read("Goal_Position"),
            "velocity_raw": bus.sync_read("Present_Velocity", normalize=False),
            "voltage_raw": bus.sync_read("Present_Voltage", normalize=False),
            "current_raw": bus.sync_read("Present_Current", normalize=False),
            "load_raw": bus.sync_read("Present_Load", normalize=False),
            "temperature_c": bus.sync_read("Present_Temperature", normalize=False),
        }
        for motor in MOTOR_NAMES:
            self.writer.writerow(
                {
                    "time_s": now - self.started_at,
                    "motor": motor,
                    "position_deg": positions[f"{motor}.pos"],
                    "goal_position_deg": values["goal_position"][motor],
                    "position_error_deg": (
                        float(values["goal_position"][motor]) - positions[f"{motor}.pos"]
                    ),
                    "velocity_raw": values["velocity_raw"][motor],
                    "voltage_v": float(values["voltage_raw"][motor]) / 10.0,
                    "current_raw": values["current_raw"][motor],
                    "torque_load_percent": float(values["load_raw"][motor]) / 10.0,
                    "temperature_c": values["temperature_c"][motor],
                }
            )
        self.file.flush()
        if now - self.last_sync_at >= 1.0:
            os.fsync(self.file.fileno())
            self.last_sync_at = now

    def close(self) -> None:
        if not self.file.closed:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()

    def print_summary(self) -> None:
        """Report sampled magnitudes; peaks are sampled peaks, not continuous maxima."""
        self.close()
        samples = {motor: [] for motor in MOTOR_NAMES}
        with self.csv_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                samples[row["motor"]].append(
                    (abs(float(row["current_raw"])), abs(float(row["torque_load_percent"])))
                )
        print("\nMotor diagnostics: full run, including standby and lowering to rest")
        print("Sample averages and peaks of absolute values; current in raw units, load estimate in %.")
        print(f"{'Motor':<18} {'Avg raw':>9} {'Peak raw':>9} {'Avg A est.':>11} {'Peak A est.':>11} {'Avg load %':>11} {'Peak load %':>12}")
        for motor, rows in samples.items():
            if not rows:
                print(f"{motor:<18} No samples")
                continue
            currents, loads = zip(*rows, strict=True)
            print(
                f"{motor:<18} {sum(currents) / len(rows):>9.2f} {max(currents):>9.2f} "
                f"{sum(currents) / len(rows) * 0.0065:>11.3f} {max(currents) * 0.0065:>11.3f} "
                f"{sum(loads) / len(rows):>11.2f} {max(loads):>12.2f}"
            )
        print("Estimated amps = raw current x 0.0065; scale not verified for this motor firmware.")
        print("Load feedback is not calibrated shaft torque (N m).")
        print(f"Raw telemetry: {self.csv_path}")

    def render_graphs(self) -> list[Path]:
        self.close()
        rows = []
        with self.csv_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                rows.append(
                    {
                        key: value if key == "motor" else float(value)
                        for key, value in row.items()
                    }
                )
        graphs = (
            ("position_deg", "time_s", "Angular position (degrees)", "Time (s)"),
            ("goal_position_deg", "time_s", "Goal position (degrees)", "Time (s)"),
            ("position_error_deg", "time_s", "Position error (degrees)", "Time (s)"),
            ("voltage_v", "time_s", "Voltage (V)", "Time (s)"),
            ("current_raw", "time_s", "Current (raw controller units)", "Time (s)"),
            ("torque_load_percent", "time_s", "Torque/load estimate (%)", "Time (s)"),
            ("velocity_raw", "time_s", "Angular velocity (raw controller units)", "Time (s)"),
            ("temperature_c", "time_s", "Temperature (°C)", "Time (s)"),
            ("voltage_v", "position_deg", "Voltage (V)", "Angular position (degrees)"),
            ("current_raw", "position_deg", "Current (raw controller units)", "Angular position (degrees)"),
            ("torque_load_percent", "position_deg", "Torque/load estimate (%)", "Angular position (degrees)"),
            ("velocity_raw", "position_deg", "Angular velocity (raw controller units)", "Angular position (degrees)"),
            ("temperature_c", "position_deg", "Temperature (°C)", "Angular position (degrees)"),
            ("position_error_deg", "position_deg", "Position error (degrees)", "Angular position (degrees)"),
        )
        paths = []
        colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
        for y_key, x_key, y_label, x_label in graphs:
            width, height = 1200, 700
            left, right, top, bottom = 100, 30, 60, 90
            plot_width = width - left - right
            plot_height = height - top - bottom
            x_values = [row[x_key] for row in rows]
            y_values = [row[y_key] for row in rows]
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)
            if x_min == x_max:
                x_max = x_min + 1
            if y_min == y_max:
                y_max = y_min + 1

            def point(row: dict) -> str:
                x = left + (row[x_key] - x_min) / (x_max - x_min) * plot_width
                y = top + (y_max - row[y_key]) / (y_max - y_min) * plot_height
                return f"{x:.2f},{y:.2f}"

            svg = [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="20">'
                f"FencingArm {y_label} vs {x_label}</text>",
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
                f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
                f'y2="{top + plot_height}" stroke="black"/>',
                f'<text x="{width / 2}" y="{height - 25}" text-anchor="middle">{x_label}</text>',
                f'<text x="20" y="{height / 2}" text-anchor="middle" '
                f'transform="rotate(-90 20 {height / 2})">{y_label}</text>',
                f'<text x="{left}" y="{top + plot_height + 25}" text-anchor="middle">{x_min:.2f}</text>',
                f'<text x="{left + plot_width}" y="{top + plot_height + 25}" '
                f'text-anchor="middle">{x_max:.2f}</text>',
                f'<text x="{left - 10}" y="{top + 5}" text-anchor="end">{y_max:.2f}</text>',
                f'<text x="{left - 10}" y="{top + plot_height + 5}" text-anchor="end">{y_min:.2f}</text>',
            ]
            for index, motor in enumerate(MOTOR_NAMES):
                motor_rows = [row for row in rows if row["motor"] == motor]
                color = colors[index]
                svg.append(
                    f'<polyline points="{" ".join(point(row) for row in motor_rows)}" '
                    f'fill="none" stroke="{color}" stroke-width="2"/>'
                )
                legend_x = left + index * 205
                svg.append(
                    f'<line x1="{legend_x}" y1="{height - 55}" x2="{legend_x + 25}" '
                    f'y2="{height - 55}" stroke="{color}" stroke-width="3"/>'
                )
                svg.append(
                    f'<text x="{legend_x + 32}" y="{height - 50}" font-size="13">{motor}</text>'
                )
            svg.append("</svg>")
            path = self.output_dir / f"{y_key}_vs_{x_key}.svg"
            path.write_text("\n".join(svg), encoding="utf-8")
            paths.append(path)
        return paths
