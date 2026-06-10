#include <Wire.h>
#include <DHT.h>

// MPU6050 Configuration
double accx, accy, accz, gyrox, gyroy, gyroz, temp;
const int Mpu = 0x68;

// DHT Sensor Configuration
#define BUZZER 12
#define DHTPIN 13
#define DHTTYPE DHT11 // Change to DHT22 if you are using a DHT22 sensor
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  
  // Initialize I2C for MPU6050
  Wire.begin();
  Wire.setClock(400000);
  delay(100);
  
  // Turn MPU on
  Wire.beginTransmission(Mpu);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission();
  delay(100);

  // Configure Gyro
  Wire.beginTransmission(Mpu);
  Wire.write(0x1B);
  Wire.write(0x00);
  Wire.endTransmission();
  delay(100);

  // Configure Accelerometer
  Wire.beginTransmission(Mpu);
  Wire.write(0x1C);
  Wire.write(0x00);
  Wire.endTransmission();
  delay(100);
  
  // Initialize DHT Sensor
  dht.begin();
  delay(100);
  
  // Initialize Serial
  Serial.begin(115200);
  delay(100);
}

void loop() {
  readIMU();
  printDataCSV();
  delay(100);
}

// Reads Data from IMU
void readIMU() {
  Wire.beginTransmission(Mpu);
  Wire.write(0x3B);
  Wire.endTransmission();
  Wire.requestFrom(Mpu, 14);
  
  accx = (int16_t)(Wire.read() << 8 | Wire.read()) / 16384.00; // g
  accy = (int16_t)(Wire.read() << 8 | Wire.read()) / 16384.00; // g
  accz = (int16_t)(Wire.read() << 8 | Wire.read()) / 16384.00; // g
  temp = (int16_t)(Wire.read() << 8 | Wire.read()) / 340.00 + 36.53; // Celsius (MPU Internal)
  gyrox = (int16_t)(Wire.read() << 8 | Wire.read()) / 131.00; // Dps
  gyroy = (int16_t)(Wire.read() << 8 | Wire.read()) / 131.00; // Dps
  gyroz = (int16_t)(Wire.read() << 8 | Wire.read()) / 131.00; // Dps
}

// Prints a concise CSV list of the data stream
// OUTPUT SCHEMA: 
// accx, accy, accz, mpu_temp, gyrox, gyroy, gyroz, dht_humidity, dht_temp
void printDataCSV() {
  // Read DHT sensor data
  float humidity = dht.readHumidity();
  float dhtTemp = dht.readTemperature(); // Celsius

  // Print MPU6050 Data
  Serial.print(accx);    Serial.print(",");
  Serial.print(accy);    Serial.print(",");
  Serial.print(accz);    Serial.print(",");
  Serial.print(temp);    Serial.print(",");
  Serial.print(gyrox);   Serial.print(",");
  Serial.print(gyroy);   Serial.print(",");
  Serial.print(gyroz);   Serial.print(",");
  
  // Print DHT Data (Checks if reading failed to avoid printing "nan")
  if (isnan(humidity)) {
    Serial.print("0.00");
  } else {
    Serial.print(humidity);
  }
  Serial.print(",");

  if (isnan(dhtTemp)) {
    Serial.print("0.01");
  } else {
    Serial.print(dhtTemp);
  }

  Serial.print(",");

  if (gyroy > 20 || gyroy < -20) {
    digitalWrite(BUZZER, HIGH);
    
    Serial.print("BUZZING");
  } else {
    digitalWrite(BUZZER, LOW);
    
    Serial.print("NOT BUZZING");
  }
  
  // Newline to finish the CSV row
  Serial.println();
}