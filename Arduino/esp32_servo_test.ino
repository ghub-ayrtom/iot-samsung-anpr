#include <ArduinoJson.h>
#include <ESP32Servo.h>
#include <HTTPClient.h>
#include <WiFi.h>

#define SERVO_PIN 4
#define BUTTON_PIN 2

const String networkName = "HolyNET";
const String networkPassword = "24100202";
const String serverURL = "http://192.168.0.107:8080/servo";

Servo servo;
int buttonState, loopCount = 0;

void setup() {
  Serial.begin(115200);

  servo.attach(SERVO_PIN);
  pinMode(BUTTON_PIN, INPUT);

  WiFi.begin(networkName, networkPassword);
  Serial.print("\nПопытка подключения к сети");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
 
  Serial.print("\nПодключение к сети с IP-адресом: ");
  Serial.print(WiFi.localIP());
  Serial.println(" прошло успешно\n");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    WiFiClient wifiClient;

    StaticJsonDocument<200> request;
    String requestSerialized, response;

    Serial.println("База данных допустимых к проезду автомобильных номеров: [A310TA73, B440TM73, K994OE73, ...]");

    request["recognized_license_plate"] = "E015HA73";

    /*

    if (loopCount % 2 == 0) {
      request["recognized_license_plate"] = "A310TA73";
    } else {
      request["recognized_license_plate"] = "E015HA73";
    }

    */

    http.begin(wifiClient, serverURL);
    http.addHeader("Content-Type", "application/json");

    serializeJson(request, requestSerialized);
    Serial.print("Распознанный автомобильный номер: ");
    Serial.println(requestSerialized + '\n');

    int httpCode = http.POST(requestSerialized);
 
    if (httpCode > 0) {
      response = http.getString();

      if (response.length() > 0) {
        Serial.println("Ответ сервера: " + response + " (" + httpCode + ")");

        if (response == "OPEN") {
          Serial.println("[АВТОМАТИЧЕСКАЯ КОМАНДА] Открыть шлагбаум...\n");
          servo.write(45);
        } else if (response == "CLOSE") {
          buttonState = digitalRead(BUTTON_PIN);

          if (buttonState == HIGH) {
            Serial.println("[РУЧНАЯ КОМАНДА] Открыть шлагбаум...\n");
            
            servo.write(15);
            delay(100);
            servo.write(90);
          }
          else {
            Serial.println("[АВТОМАТИЧЕСКАЯ КОМАНДА] Закрыть шлагбаум...\n");
            servo.write(90);
          }
        }
      }
    }
    else {
      Serial.println("Ответ сервера: " + response + " (" + httpCode + ")\n");
    }

    http.end();
  } else {
    Serial.println("\nПроизошла ошибка при попытке подключения к сети!\n");
  }

  ++loopCount;
  delay(3000);
}
