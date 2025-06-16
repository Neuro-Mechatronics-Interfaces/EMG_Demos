#include <Servo.h>

Servo myServo;
int pos = 0; // Start at neutral position

void setup() {
  Serial.begin(9600);
  myServo.attach(9);  // Attach servo to pin 9
  myServo.write(pos);
}


// [1] [Single-Threshold Logic]
// void loop() {
//   if (Serial.available() > 0) { 
//     char command = Serial.read(); 
    
//     if (command == 'W' || command == 'w') {
//       pos = 180;  // Move up
//     } 
//     else if (command == 'S' || command == 's') {
//       pos = 0;  // Move down
//     }
//     myServo.write(pos);
//   }
// }


// [2] [Proportional Logic + Single threshold Logic]
void loop() {
  if (Serial.available() > 0) { 
    int receivedAngle = Serial.parseInt(); 
    if (receivedAngle >= 0 && receivedAngle <= 180) {
      myServo.write(receivedAngle);
    }
  }
}
