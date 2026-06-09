blinkSetup() {
    pinMode(LED_BUILTIN, OUTPUT);
}

blinkLoop() {
  digitalWrite(LED_BUILTIN, HIGH);  // change state of the LED by setting the pin to the HIGH voltage level
  delay(200);                      // wait for a second
  digitalWrite(LED_BUILTIN, LOW);   // change state of the LED by setting the pin to the LOW voltage level
  delay(200); 
}