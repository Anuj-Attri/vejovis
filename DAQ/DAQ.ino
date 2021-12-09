/*

  Hardware Connections (Breakoutboard to Arduino):
  -5V = 5V (3.3V is allowed)
  -GND = GND
  -SDA = A4 (or SDA)
  -SCL = A5 (or SCL)
  -INT = Not connected

*/

#include <Wire.h>
#include "MAX30105.h"

unsigned long int red;
MAX30105 particleSensor;

#define debug Serial 


void setup()
{
  debug.begin(9600);

  // Initialize sensor
  if (particleSensor.begin() == false)
  {
    while (1);
  }

  particleSensor.setup();
}

void loop()
{
  red = particleSensor.getRed();
  red = -1*red;
  debug.println(red);

}
