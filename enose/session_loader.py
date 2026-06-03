# enose/session_loader.py
"""
Session-aware data loader for the hop e-nose dataset.

Each source CSV file in consolidation_metadata.json represents one measurement
session (the e-nose connected to the same physical sample on the same day,
producing exactly 5 cycles). This loader tracks which session each cycle belongs
to, enabling Leave-One-Session-Out (LOSO) cross-validation.

Without session tracking, a random 70/15/15 split will leak cycles from the
same physical measurement session across train and test — inflating accuracy.
"""

import os
import json
import numpy as np

from enose.data_loader import SENSOR_COLUMNS

METADATA_FILENAME = "consolidation_metadata.json"

# Maps CLASS_FILES labels (from run_hop_experiments.py) → consolidation_metadata.json keys
CLASS_TO_METADATA = {
    "Chinook_Fresh": "CHINOOK_Fresco",
    "Chinook_Aged":  "CHINOOK_Passada",
    "Comet_2025":    "COMET_2025_2",
    "Saaz_2005":     "SAAZ_2005",
    "Saaz_Fresh":    "SAAZ_Fresco",
    "Saaz_Aged":     "SAAZ_Passado",
    "Saaz_Dry":      "SAAZ_Seco",
    "Vista_Fresh":   "VISTA_Fresco",
    "Zeus_2025":     "ZEUS_2025_2",
}


def _parse_cycles_from_csv(filepath):
    """Read a consolidated CSV and return a list of 2D numpy arrays (time_steps, n_sensors)."""
    cycles = []
    current_cycle = []
    in_cycle = False

    with open(filepath, "r") as f:
        for line in f:
            if "_CICLO_INICIADO_" in line:
                in_cycle = True
                current_cycle = []
                continue
            if "_CICLO_FINALIZADO_" in line:
                if in_cycle and current_cycle:
                    cycles.append(np.array(current_cycle, dtype=np.float32))
                    current_cycle = []   # prevent re-save from duplicate marker
                in_cycle = False
                continue
            if in_cycle and len(line.strip()) > 1:
                try:
                    values = [float(s) for s in line.split()]
                    sensor_values = values[3: 3 + len(SENSOR_COLUMNS)]
                    if len(sensor_values) == len(SENSOR_COLUMNS):
                        current_cycle.append(sensor_values)
                except (ValueError, IndexError):
                    continue

    return cycles


def load_with_sessions(class_files, data_dir, sensors=None):
    """
    Load e-nose cycles and annotate each with a unique session ID.

    Session IDs are derived from consolidation_metadata.json: each source file
    (e.g. CHINOOK_Fresco_Ia.csv) is treated as one session. Cycles within the
    same source file share the same session ID.

    Args:
        class_files : dict  {class_label: filename}  — same format as CLASS_FILES
                      in run_hop_experiments.py
        data_dir    : str   path to the consolidated/ directory
        sensors     : list  sensor names to keep (default: all 7)

    Returns:
        dict with keys:
            X_raw_3d      — np.ndarray (n_cycles, time_steps, n_sensors)
            y             — np.ndarray of str class labels, shape (n_cycles,)
            session_ids   — np.ndarray of str session IDs, shape (n_cycles,)
            selected_sensors — list of sensor names actually used
            min_len       — int, time steps per cycle after truncation
            max_len       — int
            classes       — sorted list of unique class labels
            n_sessions    — int, total number of unique sessions
            sessions_per_class — dict {class_label: [session_id, ...]}
    """
    if sensors is None:
        sensors = SENSOR_COLUMNS

    # Load session metadata
    metadata_path = os.path.join(data_dir, METADATA_FILENAME)
    with open(metadata_path) as f:
        metadata = json.load(f)

    sensor_indices = [i for i, name in enumerate(SENSOR_COLUMNS) if name in sensors]
    selected_sensors = [SENSOR_COLUMNS[i] for i in sensor_indices]

    all_cycles = []      # list of 2D arrays (time_steps, n_selected_sensors)
    all_labels = []      # str class label per cycle
    all_session_ids = [] # str session ID per cycle
    sessions_per_class = {}

    for class_name, filename in class_files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"WARNING: {filepath} not found, skipping {class_name}")
            continue

        # Get session structure from metadata
        meta_key = CLASS_TO_METADATA.get(class_name)
        if meta_key and meta_key in metadata:
            cycles_per_session = metadata[meta_key]["cycles_per_file"]
            source_names = metadata[meta_key]["source_files"]
        else:
            print(f"WARNING: No metadata for '{class_name}'. Treating all cycles as one session.")
            cycles_per_session = None
            source_names = [filename]

        # Parse all cycles from the consolidated CSV
        raw_cycles = _parse_cycles_from_csv(filepath)

        # Select only the requested sensor columns
        cycles_filtered = [c[:, sensor_indices] for c in raw_cycles]

        # Build session ID for each cycle
        if cycles_per_session is not None:
            # Verify metadata matches actual cycle count
            expected = sum(cycles_per_session)
            if expected != len(raw_cycles):
                print(
                    f"WARNING: {class_name} — metadata expects {expected} cycles, "
                    f"found {len(raw_cycles)}. Falling back to groups of 5."
                )
                cycles_per_session = None

        if cycles_per_session is not None:
            session_ids_for_class = []
            for session_idx, (n_cycles_in_session, src_name) in enumerate(
                zip(cycles_per_session, source_names)
            ):
                session_id = f"{class_name}_S{session_idx:02d}"
                session_ids_for_class.extend([session_id] * n_cycles_in_session)
        else:
            # Fallback: group every 5 consecutive cycles into one session
            session_ids_for_class = [
                f"{class_name}_S{i // 5:02d}" for i in range(len(raw_cycles))
            ]

        unique_sessions_this_class = list(dict.fromkeys(session_ids_for_class))
        sessions_per_class[class_name] = unique_sessions_this_class

        all_cycles.extend(cycles_filtered)
        all_labels.extend([class_name] * len(raw_cycles))
        all_session_ids.extend(session_ids_for_class)

        print(
            f"  {class_name:20s}: {len(raw_cycles):3d} cycles | "
            f"{len(unique_sessions_this_class)} sessions "
            f"({cycles_per_session})"
        )

    if not all_cycles:
        raise ValueError("No valid cycles found in any file.")

    # Truncate all cycles to the minimum cycle length (same as existing pipeline)
    cycle_lengths = [len(c) for c in all_cycles]
    min_len = int(min(cycle_lengths))
    max_len = int(max(cycle_lengths))
    print(f"\nCycle lengths: min={min_len}, max={max_len}, mean={np.mean(cycle_lengths):.1f}")
    print(f"Truncating all cycles to {min_len} time steps.")

    standardized = [c[:min_len, :] for c in all_cycles]

    # Stack to 3D array: (n_cycles, time_steps, n_sensors)
    X_raw_3d = np.stack(standardized, axis=0).astype(np.float32)
    y = np.array(all_labels)
    session_ids = np.array(all_session_ids)

    unique_sessions = np.unique(session_ids)
    print(
        f"\nLoaded: {len(y)} cycles | {len(np.unique(y))} classes | "
        f"{len(unique_sessions)} sessions total"
    )

    return {
        "X_raw_3d": X_raw_3d,
        "y": y,
        "session_ids": session_ids,
        "selected_sensors": selected_sensors,
        "min_len": min_len,
        "max_len": max_len,
        "classes": sorted(np.unique(y).tolist()),
        "n_sessions": len(unique_sessions),
        "sessions_per_class": sessions_per_class,
    }
