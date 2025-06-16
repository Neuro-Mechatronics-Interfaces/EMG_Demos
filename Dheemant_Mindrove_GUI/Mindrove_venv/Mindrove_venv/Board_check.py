from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

BoardShim.enable_dev_board_logger() # enable logger when developing to catch relevant logs
params = MindRoveInputParams()
# params.mac_address = "0b088c"
# params.ip_address = ""
# params.ip_port = "0b088c"
board_id = BoardIds.MINDROVE_WIFI_BOARD
board_shim = BoardShim(board_id, params)

board_shim.prepare_session()
board_shim.start_stream()

eeg_channels = BoardShim.get_eeg_channels(board_id)
accel_channels = BoardShim.get_accel_channels(board_id)
sampling_rate = BoardShim.get_sampling_rate(board_id)
desc = BoardShim.get_board_descr(board_id)
dev = BoardShim.get_device_name(board_id)
window_size = 2 # seconds
num_points = window_size * sampling_rate


print(f"Board_ID:{board_id}")
print(f"board_shim: {board_shim}")
print(f"desc: {str(desc)}")
print(f"dev: {str(dev)}")

#   if board_shim.get_board_data_count() >= num_points:
#     data = board_shim.get_current_board_data(num_points)
#     eeg_data = data[eeg_channels] # output of shape (8, num_of_samples) ## Beware that depending on the electrode configuration, some channels can be *inactive*, resulting in all-zero data for that particular channel
#     accel_data = data[accel_channels] # output of shape (3, num_of_samples)
#     # process data, or print it out
#     print(eeg_data)
