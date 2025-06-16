import serial
import threading
import numpy as np
import time
import serial.tools.list_ports
import multiprocessing

# Global placeholder
arduino = None
available_ports = []

# Safe Arduino initialization function
def initialize_arduino(port='COM3', baudrate=115200):
    global arduino
    if arduino is None:
        try:
            arduino = serial.Serial(port, baudrate)
            print(f"Arduino connected on {port}")
        except Exception as e:
            print(f"Error connecting to Arduino: {e}")

# Check if we're in the main process before touching serial ports
if multiprocessing.current_process().name == "MainProcess":
    ports = serial.tools.list_ports.comports()
    for port, desc, hwid in sorted(ports):
        available_ports.append(port)
        print(f"{port}: {desc} [{hwid}]")
    
    if available_ports:
        default_port = available_ports[0]
        initialize_arduino(port=default_port)
    else:
        print("No serial ports found.")


last_motor_action_time = 0  # Track last time motor was activated
MOTOR_DELAY = 1  # Minimum delay in seconds between motor commands


class Controls_st_pc:
    def __init__(self, sensor, window_size, step_size, mvc_peak,ind_chan, percentage = 20,update_diffangle = 2.5, update_difftime = 1.5, window=None):
        self.sensor = sensor
        self.window_size = window_size
        self.step_size = step_size
        self.mvc_peak = mvc_peak
        self.percentage = percentage
        self.update_diffangle = update_diffangle
        self.update_difftime = update_difftime
        self.window = window
        self.ind_chan = ind_chan

    def Threshold_inference(self):
        """
        Compute threshold bounds from MVC.
        """
        thresholds = []
        upper_bounds = []
        lower_bounds = []
        
        for ch in range(self.sensor.num_channels):
            thresholds.append((self.window.current_percentage / 100) * 1)
            upper_bounds.append(((self.window.current_percentage + 2.5) / 100) * 1)
            lower_bounds.append(((self.window.current_percentage - 2.5) / 100) * 1)
        
        
        return thresholds, upper_bounds, lower_bounds

    def Proportional_inference(self, data_x, in_min, in_max, out_min, out_max):
        """
        Compute Proportional bounds, Maps a number from one range to another..
        """
        return (data_x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def execute_motor_action(self, command):
        def send_command():
            global last_motor_action_time
            current_time = time.time()
            if arduino is None: 
                return
            if current_time - last_motor_action_time > self.update_difftime:
                arduino.write(command.encode())  # Convert to bytes
                arduino.flush()
                time.sleep(self.update_difftime)  # Prevent overwhelming Arduino
                last_motor_action_time = current_time

        threading.Thread(target=send_command, daemon=True).start()
        
    def execute_motor_action_new(self, command):
        """Sends a motor command to Arduino with rate-limiting to prevent flooding."""
        if arduino is None:
            return
        if not hasattr(self, "last_motor_action_time"):
            self.last_motor_action_time = 0  # Initialize if not already set

        current_time = time.time()

        # Rate-limit to 1 update per second
        if current_time - self.last_motor_action_time > self.update_difftime:
            def send_command():
                try:
                    arduino.write(command.encode())  # Convert to bytes
                    arduino.flush()
                    self.last_motor_action_time = time.time()  # Update last action time
                except Exception as e:
                    print(f"Error sending command to Arduino: {e}")

            threading.Thread(target=send_command, daemon=True).start()

    def Thresh_Implement_old(self, window_data, percentage):
        """
        Apply threshold logic to detect movement onset.
        """
        # print(f"data: {window_data}")
        # print(f"data: {len(window_data)}")
        window_means = [np.mean(ch_data) for ch_data in window_data]
        # print(window_means)
        # window_means = np.mean(window_data)
        thresholds, upper_bounds, lower_bounds = self.Threshold_inference(percentage, self.mvc_peak)

        all_below = all(wm <= lb for wm, lb in zip(window_means, lower_bounds))
        all_within_lower = all(lb <= wm <= t for wm, lb, t in zip(window_means, lower_bounds, thresholds))
        all_within_upper = all(t <= wm <= ub for wm, t, ub in zip(window_means, thresholds, upper_bounds))
        all_above = all(wm >= ub for wm, ub in zip(window_means, upper_bounds))

        if all_below:
            # print("Movement stopped across all channels.")
            self.execute_motor_action('s')
            return "Down"
        
        # elif all_within_lower:
        #     print("Slightly under the threshold across all channels.")
            
        # elif all_within_upper:
        #     print("Slightly above the threshold across all channels.")
            
        elif all_above:
            # print("Movement onset detected across all channels.")
            self.execute_motor_action('w')
            return "Up"
        
    def Thresh_Implement(self, window_data):
        # Compute means from the window data (assume window_data is a list of arrays)
        window_means = [np.mean(ch_data) for ch_data in window_data]
        # print(window_means)
        thresholds, upper_bounds, lower_bounds = self.Threshold_inference()
        
        # Ensure thresholds, upper_bounds, lower_bounds are iterable.
        # If they are not, assume we want the same value for all channels.
        num_channels = len(window_means)
        
        if not hasattr(thresholds, '__iter__'):
            thresholds = [thresholds] * num_channels
        if not hasattr(upper_bounds, '__iter__'):
            upper_bounds = [upper_bounds] * num_channels
        if not hasattr(lower_bounds, '__iter__'):
            lower_bounds = [lower_bounds] * num_channels

        all_below = all(wm <= lb for wm, lb in zip(window_means, lower_bounds))
        all_within_lower = all(lb <= wm <= t for wm, lb, t in zip(window_means, lower_bounds, thresholds))
        all_within_upper = all(t <= wm <= ub for wm, t, ub in zip(window_means, thresholds, upper_bounds))
        all_above = all(wm >= ub for wm, ub in zip(window_means, upper_bounds))

        if all_below:
            self.execute_motor_action_new('0')
            print(f"Avg Activation: {'Below Thresh'}, Target Angle: {'0'}")
            return "Down"
        elif all_above:
            self.execute_motor_action_new('180')
            print(f"Avg Activation: {'Above Thresh'}, Target Angle: {'180'}")
            return "Up"
        
        
        
    # def Proportional_Implement(self, window_data, in_min=0, in_max=1, out_min=0, out_max=180):   # Version 1
    #     """
    #     Apply proportional control logic to control servo angle with rate-limiting.
    #     """
    #     window_means = [np.mean(ch_data) for ch_data in window_data]
          
    #     top_means = sorted(window_means, reverse=True)[:4]  # Select the 4 highest means
    #     avg_activation = np.mean(top_means)  # Compute their average
        
    #     target_angle = self.Proportional_inference(avg_activation, in_min, in_max, out_min, out_max)  # Map to angle
    #     target_angle = max(out_min, min(out_max, int(target_angle)))  # Clip to valid servo range
        
    #     angle_tolerance = self.window.current_angle_tolerance
    #     time_tolerance = self.window.current_time_tolerance

    #     # Rate-limiting: Only send if the change is greater than 2 degrees
    #     if not hasattr(self, 'last_sent_angle'):  # Initialize if not set
    #         self.last_sent_angle = target_angle
    #     if not hasattr(self, 'last_sent_time'):
    #         self.last_sent_time = time.time()

    #     current_time = time.time()

    #     if abs(target_angle - self.last_sent_angle) > angle_tolerance and (current_time - self.last_sent_time > time_tolerance):  # Only update if significant change
    #         print(f"Avg Activation: {avg_activation}, Target Angle: {target_angle}")
    #         # self.execute_motor_action(str(target_angle))  # Send angle to Arduino    
    #         self.execute_motor_action(f'set_gripper:{abs(target_angle-180)};')    # Jonathan's Code Integration
    #         self.last_sent_angle = target_angle  # Update last sent angle
    #         self.last_sent_time = current_time  # Update last sent time
        
    #     return target_angle
    
    def Proportional_Implement_v2(self, window_data,ind_chan, in_min=0, in_max=1, out_min=0, out_max=180):   # Version 2
        """
        Apply proportional control logic to control servo angle with rate-limiting.
        """
        window_means = [np.mean(ch_data) for ch_data in window_data]
        Selected_chan_activation = window_means[ind_chan] 
        
        target_angle = self.Proportional_inference(Selected_chan_activation, in_min, in_max, out_min, out_max)  # Map to angle
        target_angle = max(out_min, min(out_max, int(target_angle)))  # Clip to valid servo range
        
        angle_tolerance = self.window.current_angle_tolerance
        time_tolerance = self.window.current_time_tolerance

        # Rate-limiting: Only send if the change is greater than 2 degrees
        if not hasattr(self, 'last_sent_angle'):  # Initialize if not set
            self.last_sent_angle = target_angle
        if not hasattr(self, 'last_sent_time'):
            self.last_sent_time = time.time()

        current_time = time.time()

        if abs(target_angle - self.last_sent_angle) > angle_tolerance and (current_time - self.last_sent_time > time_tolerance):  # Only update if significant change
            print(f"Avg Activation {ind_chan+1}: {Selected_chan_activation}, Target Angle: {target_angle}")
            # self.execute_motor_action(str(target_angle))  # Send angle to Arduino    
            self.execute_motor_action(f'set_gripper:{abs(target_angle-180)};')    # Jonathan's Code Integration
            self.last_sent_angle = target_angle  # Update last sent angle
            self.last_sent_time = current_time  # Update last sent time
        
        return target_angle
    
    
    def Proportional_Implement_v3(self, window_data,ind_chan, in_min=0, in_max=1, out_min=0, out_max=180):   # Version 3 (Under Dev)
        """
        Apply proportional control logic to control servo angle with rate-limiting.
        """
        window_means = [np.mean(ch_data) for ch_data in window_data]
        Selected_chan_activation = window_means[ind_chan] 
        
        target_angle = self.Proportional_inference(Selected_chan_activation, in_min, in_max, out_min, out_max)  # Map to angle
        target_angle = max(out_min, min(out_max, int(target_angle)))  # Clip to valid servo range
        
        angle_tolerance = self.window.current_angle_tolerance
        time_tolerance = self.window.current_time_tolerance

        # Rate-limiting: Only send if the change is greater than 2 degrees
        if not hasattr(self, 'last_sent_angle'):  # Initialize if not set
            self.last_sent_angle = target_angle
        if not hasattr(self, 'last_sent_time'):
            self.last_sent_time = time.time()

        current_time = time.time()

        if abs(target_angle - self.last_sent_angle) > angle_tolerance and (current_time - self.last_sent_time > time_tolerance):  # Only update if significant change
            print(f"Avg Activation {ind_chan+1}: {Selected_chan_activation}, Target Angle: {target_angle}")
            # self.execute_motor_action(str(target_angle))  # Send angle to Arduino    
            # self.execute_motor_action(f'set_gripper:{abs(target_angle-180)}:movemotor:2:{abs(target_angle-180)};')    # Jonathan's Code Integration
            self.execute_motor_action(f'movemotor:1:{abs(target_angle-180)};')    # Jonathan's Code Integration

            self.last_sent_angle = target_angle  # Update last sent angle
            self.last_sent_time = current_time  # Update last sent time
        
        return target_angle
    
    
    def Proportional_Implement_v4(self, window_data,ind_chan, in_min=0, in_max=1, out_min=0, out_max=180, command = "set_gripper:{};"):   # Version 3 (Under Dev)
        """
        Apply proportional control logic to control servo angle with rate-limiting.
        """
        window_means = [np.mean(ch_data) for ch_data in window_data]
        Selected_chan_activation = window_means[ind_chan] 
        
        target_angle = self.Proportional_inference(Selected_chan_activation, in_min, in_max, out_min, out_max)  # Map to angle
        target_angle = max(out_min, min(out_max, int(target_angle)))  # Clip to valid servo range
        
        angle_tolerance = self.window.current_angle_tolerance
        time_tolerance = self.window.current_time_tolerance

        # Rate-limiting: Only send if the change is greater than 2 degrees
        if not hasattr(self, 'last_sent_angle'):  # Initialize if not set
            self.last_sent_angle = target_angle
        if not hasattr(self, 'last_sent_time'):
            self.last_sent_time = time.time()

        current_time = time.time()

        if abs(target_angle - self.last_sent_angle) > angle_tolerance and (current_time - self.last_sent_time > time_tolerance):  # Only update if significant change
            print(f"Avg Activation {ind_chan+1}: {Selected_chan_activation}, Target Angle: {target_angle}")
            
            
            if command == "set_gripper:{};":
                print(str(command.format(abs(target_angle-180))))
                self.execute_motor_action(str(command.format(abs(target_angle-180))))
                
            else:
                print(str(command.format(abs(target_angle-90))))
                self.execute_motor_action(str(command.format(abs(target_angle-90))))
                

            # self.execute_motor_action(str(target_angle))  # Send angle to Arduino    
            # self.execute_motor_action(f'set_gripper:{abs(target_angle-180)};')    # Jonathan's Code Integration
            # self.execute_motor_action(f'movemotor:1:{abs(target_angle-180)};')    # Jonathan's Code Integration

            self.last_sent_angle = target_angle  # Update last sent angle
            self.last_sent_time = current_time  # Update last sent time
        
        return target_angle
    
    def Proportional_Implement_v5(self, window_data,ind_chan, in_min=0, in_max=1, out_min=0, out_max=180, command = "set_gripper:{};", Force_channel = None):   # Version 3 (Under Dev)
        """
        Apply proportional control logic to control servo angle with rate-limiting.
        """
        
        if Force_channel == None or Force_channel == "None":
            window_means = [np.mean(ch_data) for ch_data in window_data]
            Selected_chan_activation = window_means[ind_chan] 
            
            target_angle = self.Proportional_inference(Selected_chan_activation, in_min, in_max, out_min, out_max)  # Map to angle
            target_angle = max(out_min, min(out_max, int(target_angle)))  # Clip to valid servo range
            
            angle_tolerance = self.window.current_angle_tolerance
            time_tolerance = self.window.current_time_tolerance

            # Rate-limiting: Only send if the change is greater than 2 degrees
            if not hasattr(self, 'last_sent_angle'):  # Initialize if not set
                self.last_sent_angle = target_angle
            if not hasattr(self, 'last_sent_time'):
                self.last_sent_time = time.time()

            current_time = time.time()

            if abs(target_angle - self.last_sent_angle) > angle_tolerance and (current_time - self.last_sent_time > time_tolerance):  # Only update if significant change
                print(f"Avg Activation {ind_chan+1}: {Selected_chan_activation}, Target Angle: {target_angle}")
                
                
                if command == "set_gripper:{};":
                    print(str(command.format(abs(target_angle-180))))
                    self.execute_motor_action(str(command.format(abs(target_angle-180))))
                    
                else:
                    print(str(command.format(abs(target_angle-90))))
                    self.execute_motor_action(str(command.format(abs(target_angle-90))))
                    

                # self.execute_motor_action(str(target_angle))  # Send angle to Arduino    
                # self.execute_motor_action(f'set_gripper:{abs(target_angle-180)};')    # Jonathan's Code Integration
                # self.execute_motor_action(f'movemotor:1:{abs(target_angle-180)};')    # Jonathan's Code Integration

                self.last_sent_angle = target_angle  # Update last sent angle
                self.last_sent_time = current_time  # Update last sent time
                
        else:
            window_means = [np.mean(ch_data) for ch_data in window_data]
            Selected_chan_activation = window_means[int(Force_channel)-1] 
            
            target_angle = self.Proportional_inference(Selected_chan_activation, in_min, in_max, out_min, out_max)  # Map to angle
            target_angle = max(out_min, min(out_max, int(target_angle)))  # Clip to valid servo range
            
            angle_tolerance = self.window.current_angle_tolerance
            time_tolerance = self.window.current_time_tolerance

            # Rate-limiting: Only send if the change is greater than 2 degrees
            if not hasattr(self, 'last_sent_angle'):  # Initialize if not set
                self.last_sent_angle = target_angle
            if not hasattr(self, 'last_sent_time'):
                self.last_sent_time = time.time()

            current_time = time.time()

            if abs(target_angle - self.last_sent_angle) > angle_tolerance and (current_time - self.last_sent_time > time_tolerance):  # Only update if significant change
                print(f"Avg Activation {int(Force_channel)}: {Selected_chan_activation}, Target Angle: {target_angle}")
                
                
                if command == "set_gripper:{};":
                    print(str(command.format(abs(target_angle-180))))
                    self.execute_motor_action(str(command.format(abs(target_angle-180))))
                    
                else:
                    print(str(command.format(abs(target_angle-90))))
                    self.execute_motor_action(str(command.format(abs(target_angle-90))))
                    

                # self.execute_motor_action(str(target_angle))  # Send angle to Arduino    
                # self.execute_motor_action(f'set_gripper:{abs(target_angle-180)};')    # Jonathan's Code Integration
                # self.execute_motor_action(f'movemotor:1:{abs(target_angle-180)};')    # Jonathan's Code Integration

                self.last_sent_angle = target_angle  # Update last sent angle
                self.last_sent_time = current_time  # Update last sent time
        
        return target_angle

