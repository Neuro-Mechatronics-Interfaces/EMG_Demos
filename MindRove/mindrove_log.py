"""
Script that logs data from the MindRove band, specifically EMG and IMU data.
The user specified a duration for the recording, and the script saves the data to a CSV file.

Author: Jonathan Shulgach
Date Created: 06/11/2025
"""

import time
import csv
import argparse
from mindrove.board_shim import BoardShim, BoardIds, MindRoveInputParams

def record_emg_imu(duration_sec, n_emg_channels=8):
    # === Setup MindRove Board ===
    params = MindRoveInputParams()
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    board = BoardShim(board_id, params)

    # === Get channel indices ===
    emg_channels = board.get_exg_channels(board_id)[:n_emg_channels]
    accel_channels = board.get_accel_channels(board_id)
    gyro_channels = board.get_gyro_channels(board_id)

    print(f"Using EMG channels: {emg_channels}")
    print(f"Using Accelerometer channels: {accel_channels}")
    print(f"Using Gyroscope channels: {gyro_channels}")

    # === Combine channels ===
    all_channels = emg_channels + accel_channels + gyro_channels
    channel_names = (
        [f'EMG_{i}' for i in range(len(emg_channels))] +
        [f'ACCEL_{i}' for i in range(len(accel_channels))] +
        [f'GYRO_{i}' for i in range(len(gyro_channels))]
    )

    print(f"Recording EMG and IMU data for {duration_sec} seconds...")
    print(f"Channels being recorded: {channel_names}")

    # === Start data stream ===
    board.prepare_session()
    board.start_stream()
    time.sleep(duration_sec)
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()
    print("Recording stopped.")

    # === Extract channel data ===
    recorded_data = data[all_channels]
    n_samples = recorded_data.shape[1]
    sample_rate = board.get_sampling_rate(board_id)

    # === Create elapsed time in milliseconds
    elapsed_ms = [int((i / sample_rate) * 1000) for i in range(n_samples)]

    # === Save to CSV ===
    output_csv = f"data_{int(time.time())}.csv"
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['elapsed_ms'] + channel_names)
        for i in range(n_samples):
            row = [elapsed_ms[i]] + [float(f"{x:.6f}") for x in recorded_data[:, i]]
            writer.writerow(row)

    print(f"Saved {n_samples} samples to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record raw EMG and IMU data from MindRove band.")
    parser.add_argument('--duration', type=int, default=1, help='Duration of recording in seconds')
    args = parser.parse_args()

    record_emg_imu(args.duration, 8)
