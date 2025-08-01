import time
from mindrove_playback import MindRovePlaybackClient

#file_path = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_05_07\raw\WristExtension_002"
file_path = r"C:\Users\HP\Documents\Github\EMG_Demos\MindRove\data\EMG_Smart_select_FRing_data_2025_06_02_Dheemant_Actual_0_unfilt.csv"
client = MindRovePlaybackClient(file_path, block_size=50, loopback=True, enable_lsl=True)
client.start_streaming()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    client.stop_streaming()