"""
realtime_playback.py

This script allows you to play back EMG data from a CSV file using the MindRove Board API,
and view it in a scrolling plot. Before playback, it checks and fixes the input file format
to make sure it's compatible with MindRove's requirements.

To use:
1. Put your EMG CSV data file in the `data/` directory.
2. Change `file_path` below to your data file name, or leave blank for a filedialogue.
3. Run the script!

Author: Jonathan Shulgach

"""
from tkinter import filedialog
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from MindRove.utilities.enforce_mindrove_format import enforce_mindrove_format
from MindRove.utilities.scrolling_emg_plot import ScrollingEMGPlot

# Enable detailed logging for debugging (can turn off in production)
BoardShim.enable_dev_board_logger()


def setup_playback_board(file_path, original_board_id=BoardIds.MINDROVE_WIFI_BOARD, loopback=False):
    """
    Helper function to set up the playback board with a file.

    Args:
        file_path (str): Path to the playback file.
        original_board_id (int): Original board ID for MindRove.
        loopback (bool): Whether to enable loopback mode.

    Returns:
        BoardShim: Configured BoardShim instance for playback.

    """

    # Set up parameters for playback mode
    params = MindRoveInputParams()
    params.master_board = original_board_id
    params.other_info = str(original_board_id).zfill(32)  # Required by API
    params.file = file_path

    # Choose playback board type
    playback_id = BoardIds.PLAYBACK_FILE_BOARD

    # Initialize board for playback
    board = BoardShim(playback_id, params)
    board.prepare_session()

    # Optional: Enable loopback mode
    if loopback:
        board.config_board('loopback_true')  # Enable this for loopback mode

    return board


def open_file(file_path=None):
    if file_path is None:
        # Open a file dialog to select the file
        file_path = filedialog.askopenfilename(
            title="Select EMG Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            print("No file selected. Exiting.")
            exit(1)

    return file_path


if __name__ == "__main__":

    # 1. Set your data file
    file_path = "MindRove/data/EMG_Smart_select_FRing_data_2025_06_02_Dheemant_Actual_0_unfilt.csv"
    file_path = open_file(file_path)  # Or pass file_path directly
    #file_path = open_file()  # Or pass file_path directly

    # 2. Ensure file format is valid
    # The MindRove API expects tab-delimited files with 32 columns, including a timestamp column.
    # This function will:
    #   - Convert to tab-delimited if needed
    #   - Add a timestamp column if missing (using UNIX time)
    #   - Pad to 32 columns if needed
    # Set `delimiter` to match your original file (',' for CSV, '\t' for tab-delimited).
    enforce_mindrove_format(
        file_path,
        delimiter='\t',  # Original delimiter in your data file
        new_delimiter='\t',  # Output delimiter required by MindRove
        has_headers=True  # Set to True if your file includes a header row
    )

    # 3. Set up MindRove for playback
    board = setup_playback_board(file_path, loopback=False)

    # 4. Run the scrolling EMG plotter
    plotter = ScrollingEMGPlot(
        board=board,
        n_channels=4,         # Number of EMG channels to display (adjust to match your data)
        duration_sec=2,       # How many seconds to show in the scrolling window
        band=(70, 400),       # Bandpass filter range in Hz (change as needed)
        y_range=(-500, 500),  # Vertical axis range for the plot
    )
    plotter.run()
