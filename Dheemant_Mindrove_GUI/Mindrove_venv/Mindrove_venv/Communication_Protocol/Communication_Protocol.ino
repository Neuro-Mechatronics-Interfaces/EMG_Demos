// -------------------------------------------------------------------------------------------------

// Code for Communication protocol for Exoskeleton
// Written By: Dheemant Jallepalli
// NeuroMechatronics Lab

// -------------------------------------------------------------------------------------------------

#include <Servo.h>

String inputString = "";      // String to hold incoming data
boolean stringComplete = false;  // Whether the string is complete

// Create servo objects for each servo
Servo handServos[4];  // 4 servos for "hand"
Servo wristServos[2];  // 2 servos for "wrist"

// Define your pins here
int Hand_Servo1_Pin = 3;
int Hand_Servo2_Pin = 5;
int Hand_Servo3_Pin = 6;
int Hand_Servo4_Pin = 9;
int Wrist_Servo1_Pin = 10;
int Wrist_Servo2_Pin = 11;

const char* elementNames[] = {"hand", "wrist"};
const int elementPins[][4] = {
  {Hand_Servo1_Pin, Hand_Servo2_Pin, Hand_Servo3_Pin, Hand_Servo4_Pin},  // hand
  {Wrist_Servo1_Pin, Wrist_Servo2_Pin, -1, -1}  // wrist 
};
const int numElements = 2;  
const int elementPinCount[] = {4, 2};  // hand has 4 pins, wrist has 2 pins

void setup() {
  // Attach all servos
  // Hand servos
  for (int i = 0; i < 4; i++) {
    handServos[i].attach(elementPins[0][i]);
    handServos[i].write(90);  // Initialize to middle position
  }
  
  // Wrist servos
  for (int i = 0; i < 2; i++) {
    wristServos[i].attach(elementPins[1][i]);
    wristServos[i].write(90);  // Initialize to middle position
  }

  Serial.begin(9600);
  inputString.reserve(200);
  
  Serial.println("Arduino ready to receive servo commands");
  Serial.println("Format: element:[values]; (e.g., hand:[0,90,180,45]; wrist:[180,0];)");
  Serial.println("Values should be 0-180 for servo positions");
}

void loop() {
  // Process complete commands
  if (stringComplete) {
    processCommandString(inputString);
    
    // Clear the string for new input
    inputString = "";
    stringComplete = false;
  }
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    inputString += inChar;
    
    // If the incoming character is a semicolon, set a flag
    if (inChar == ';') {
      stringComplete = true;
    }
  }
}

void processCommandString(String commandString) {
  // Process each command in the string
  int startPos = 0;
  int endPos = 0;
  
  while (startPos < commandString.length()) {
    // Find the next semicolon
    endPos = commandString.indexOf(';', startPos);
    
    if (endPos == -1) break;  // No more semicolons found
    
    String command = commandString.substring(startPos, endPos + 1);
    processCommand(command);
    
    // Move to the next command
    startPos = endPos + 1;
  }
}

void processCommand(String command) {
  // Find position of key elements
  int colonPos = command.indexOf(':');
  int semicolonPos = command.indexOf(';');
  
  if (colonPos == -1 || semicolonPos == -1) {
    Serial.println("Invalid command format");
    return;
  }
  
  // Extract element name and values
  String elementName = command.substring(0, colonPos);
  String valueStr = command.substring(colonPos + 1, semicolonPos);
  
  // Process the command if it's in the format [x,x,x,...]
  if (valueStr.startsWith("[") && valueStr.endsWith("]")) {
    valueStr = valueStr.substring(1, valueStr.length() - 1);
    
    // Try to match the element name to our known elements
    int elementIndex = -1;
    for (int i = 0; i < numElements; i++) {
      if (elementName.equals(elementNames[i])) {
        elementIndex = i;
        break;
      }
    }
    
    if (elementIndex != -1) {
      // Parse the values
      int values[4] = {0, 0, 0, 0};  // Initialize with middle position (90 degrees)
      int valueIndex = 0;
      int commaPos = -1;
      int startPos = 0;
      
      // Parse each value between commas
      while (valueIndex < 4 && startPos < valueStr.length()) {
        commaPos = valueStr.indexOf(',', startPos);
        
        if (commaPos == -1) {
          // Last value
          values[valueIndex] = valueStr.substring(startPos).toInt();
          valueIndex++;
          break;
        } else {
          values[valueIndex] = valueStr.substring(startPos, commaPos).toInt();
          startPos = commaPos + 1;
          valueIndex++;
        }
      }
      
      // Set the servo positions according to the values
      for (int i = 0; i < min(valueIndex, elementPinCount[elementIndex]); i++) {
        // Constrain values to valid servo range (0-180)
        int servoPos = constrain(values[i], 0, 180);
        
        // Set servo positions based on which element we're controlling
        if (elementIndex == 0) {  // hand
          handServos[i].write(servoPos);
        } else if (elementIndex == 1) {  // wrist
          wristServos[i].write(servoPos);
        }
      }
      
      // Construct response
      String response = elementName + " servos set to: [";
      for (int i = 0; i < min(valueIndex, elementPinCount[elementIndex]); i++) {
        response += String(values[i]);
        if (i < min(valueIndex, elementPinCount[elementIndex]) - 1) {
          response += ",";
        }
      }
      response += "]";
      
      Serial.println(response);
    } else {
      Serial.println("Unknown element: " + elementName);
    }
  } else {
    Serial.println("Invalid value format for " + elementName);
  }
}