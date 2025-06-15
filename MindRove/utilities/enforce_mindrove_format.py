import numpy as np
import time


def enforce_mindrove_format(file_path, delimiter='\t', new_delimiter=None, has_headers=False, fs=500, start_unix_time=None):
    """
    Ensure the MindRove file has 32 columns and a valid timestamp channel (col 28/index 27).
    Adds a timestamp column if missing. Pads extra columns with zeros if needed.

    Parameters:
        file_path (str): Path to the input file.
        delimiter (str): Delimiter used in the input file.
        new_delimiter (str): New delimiter to use when saving the file.
        has_headers (bool): Whether the input file has a header row.
        fs (int): Sampling frequency in Hz (default is 500).
        start_unix_time (float): UNIX time to use as the start for timestamping (default is current time).

    Returns:
        None

    """
    # Read as tab-delimited
    skiprows = 1 if has_headers else 0
    data = np.loadtxt(file_path, delimiter=delimiter, skiprows=skiprows)

    n_rows, n_cols = data.shape
    print(f"Original shape: {data.shape}")

    if n_cols < 28:
        # Pad with zeros to ensure at least 28 columns
        data = np.hstack([data, np.zeros((n_rows, 28 - n_cols))])

        # Build timestamp vector (milliseconds since start)
        print("No timestamp column present. Adding it at column 28 (index 27).")
        if start_unix_time is None:
            # Use the current UNIX time as the start
            start_unix_time = time.time()

        #timestamp_col = np.arange(n_rows) * (1000.0 / fs)  # ms at 500Hz
        timestamp_col = start_unix_time + np.arange(n_rows) / fs
        # Insert timestamp as the 28th column
        data = np.insert(data, 27, timestamp_col, axis=1)
        n_cols = data.shape[1]
    else:
        # Validate timestamp column? (Optional)
        print("Timestamp column found. (No validation performed.)")

    # Pad up to 32 columns if needed
    if n_cols < 32:
        data = np.hstack([data, np.zeros((n_rows, 32 - n_cols))])
        print(f"Padded to {data.shape[1]} columns.")

    # Save with original delimiter
    if new_delimiter is not None:
        delimiter = new_delimiter

    np.savetxt(file_path, data, delimiter=delimiter, fmt='%.6f')
    print(f"File {file_path} adjusted to MindRove format with new delimiter: {delimiter}")

