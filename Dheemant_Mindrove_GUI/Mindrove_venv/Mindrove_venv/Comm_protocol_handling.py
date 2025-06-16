# -------------------------------------------------------------------------------------------------

# Code for Communication protocol for Exoskeleton
# Written By: Dheemant Jallepallihand:[]\
# NeuroMechatronics Lab

# -------------------------------------------------------------------------------------------------


import serial
import time
import re

class ArduinoController:
    def __init__(self, port, baud_rate=9600, timeout=1):
        """Initialize the serial connection to Arduino"""
        try:
            self.ser = serial.Serial(port, baud_rate, timeout=timeout)
            time.sleep(2)  
        except serial.SerialException as e:
            print(f"Error connecting to Arduino: {e}")
            raise
        
    def send_command_string(self, command_string):
        """Send a string containing multiple commands to Arduino"""
        # Strip spaces from the command string
        command_string = command_string.replace(" ", "")

        print(f"Sending: {command_string}")
        self.ser.write(command_string.encode())
        time.sleep(0.2)  
        
        # Read response from Arduino
        self._read_response()
    
    def _read_response(self):
        """Read and print response from Arduino"""
        response = ""
        start_time = time.time()
        
        # Read for up to 1 second or until no more data
        while (time.time() - start_time < 1) or self.ser.in_waiting > 0:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    print(f"Arduino: {line}")
                    response += line + "\n"
            else:
                time.sleep(0.1)
                if self.ser.in_waiting == 0:
                    break
        
        return response
    
    def close(self):
        """Close the serial connection"""
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
            print("Connection to Arduino closed")


def is_valid_command_string(cmd_string):
    """Basic validation for command string format"""
    # Remove spaces for validation
    cmd_string = cmd_string.replace(" ", "")
    
    # Check if there's at least one properly formatted element
    pattern = r'[a-zA-Z]+:\[\d+(?:,\d+)*\];'
    return bool(re.search(pattern, cmd_string))


def start_command_interface(port):
    """Start interactive command interface"""
    try:
        arduino = ArduinoController(port)
        
        print("\n=== Arduino Command Interface ===")
        print("Enter commands in the format: 'element:[values];'")
        print("Multiple commands can be combined: 'hand:[1,0,0,0]; elbow:[0,1,1]; something_else:[1,0];'")
        print("Type 'exit' to quit\n")
        
        while True:
            user_input = input("Enter command(s): ")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
                
            if not user_input.strip():
                continue
                
            if is_valid_command_string(user_input):
                arduino.send_command_string(user_input)
            else:
                print("Invalid command format. Please use: element:[values];")
                print("Example: hand:[1,0,0,0]; elbow:[0,1,1];")
        
        arduino.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    PORT = 'COM9'

    start_command_interface(PORT)