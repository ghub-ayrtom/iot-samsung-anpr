#include "Arduino.h"
#include "esp_camera.h"
#include <ESP32Servo.h>
#include <HTTPClient.h>
#include <WiFi.h>

#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
#define SERVO_PIN_NUM 14
#define BUTTON_PIN_NUM 15

const String networkName = "HolyNET";
const String networkPassword = "24100202";

HTTPClient http;
const String serverURL = "http://192.168.0.107:8080/recognize";

Servo servo;
int buttonState = 0;

void setup()
{
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  servo.attach(SERVO_PIN_NUM);
  servo.write(0);
  pinMode(BUTTON_PIN_NUM, INPUT);

  WiFi.begin(networkName, networkPassword);
  Serial.print("Attempting to connect to the network");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
 
  Serial.print("\nNetwork connection with IP address: ");
  Serial.print(WiFi.localIP());
  Serial.println(" was successful");

  camera_config_t config;

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_VGA;
  config.pixel_format = PIXFORMAT_JPEG; // PIXFORMAT_RGB565 for detection/recognition
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 4;
  config.fb_count = 1;

  /*

  if (config.pixel_format == PIXFORMAT_JPEG) {
    if (psramFound()) {
      config.jpeg_quality = 10;
      config.fb_count = 2;
      config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
      config.frame_size = FRAMESIZE_SVGA;
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    config.frame_size = FRAMESIZE_240X240;

    #if CONFIG_IDF_TARGET_ESP32S3
      config.fb_count = 2;
    #endif
  }

  */

  #if defined(CAMERA_MODEL_ESP_EYE)
    pinMode(13, INPUT_PULLUP);
    pinMode(14, INPUT_PULLUP);
  #endif

  esp_err_t error = esp_camera_init(&config);

  if (error != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x!\n", error);
    return;
  }

  sensor_t * sensor = esp_camera_sensor_get();

  sensor->set_brightness(sensor, 0);    // -2 to 2
  sensor->set_contrast(sensor, 0);       // -2 to 2
  sensor->set_saturation(sensor, 0);     // -2 to 2
  sensor->set_special_effect(sensor, 0); // 0 to 6 (0 - No Effect, 1 - Negative, 2 - Grayscale, 3 - Red Tint, 4 - Green Tint, 5 - Blue Tint, 6 - Sepia)
  sensor->set_whitebal(sensor, 1);       // 0 = disable, 1 = enable
  sensor->set_awb_gain(sensor, 1);       // 0 = disable, 1 = enable
  sensor->set_wb_mode(sensor, 0);        // 0 to 4 - if awb_gain enabled (0 - Auto, 1 - Sunny, 2 - Cloudy, 3 - Office, 4 - Home)
  sensor->set_exposure_ctrl(sensor, 1);  // 0 = disable, 1 = enable
  sensor->set_aec2(sensor, 1);           // 0 = disable, 1 = enable
  sensor->set_ae_level(sensor, 0);       // -2 to 2
  sensor->set_aec_value(sensor, 300);    // 0 to 1200
  sensor->set_gain_ctrl(sensor, 1);      // 0 = disable, 1 = enable
  sensor->set_agc_gain(sensor, 0);       // 0 to 30
  sensor->set_gainceiling(sensor, (gainceiling_t)0);  // 0 to 6
  sensor->set_bpc(sensor, 0);            // 0 = disable, 1 = enable
  sensor->set_wpc(sensor, 1);            // 0 = disable, 1 = enable
  sensor->set_raw_gma(sensor, 1);        // 0 = disable, 1 = enable
  sensor->set_lenc(sensor, 1);           // 0 = disable, 1 = enable
  sensor->set_hmirror(sensor, 0);        // 0 = disable, 1 = enable
  sensor->set_vflip(sensor, 0);          // 0 = disable, 1 = enable
  sensor->set_dcw(sensor, 1);            // 0 = disable, 1 = enable
  sensor->set_colorbar(sensor, 0);       // 0 = disable, 1 = enable

  if (sensor->id.PID == OV3660_PID) {
    sensor->set_vflip(sensor, 1);
    sensor->set_brightness(sensor, 1);
    sensor->set_saturation(sensor, -2);
  }

  #if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
    sensor->set_vflip(sensor, 1);
    sensor->set_hmirror(sensor, 1);
  #endif

  #if defined(CAMERA_MODEL_ESP32S3_EYE)
    sensor->set_vflip(sensor, 1);
  #endif

  #if defined(LED_GPIO_NUM)
    setupLedFlash(LED_GPIO_NUM);
  #endif
}

void loop()
{
  camera_fb_t *frame_buffer = NULL;
  frame_buffer = esp_camera_fb_get();

  if (!frame_buffer) {
    Serial.println("Camera capture failed!\n");
    return;
  }

  if (WiFi.status() == WL_CONNECTED) {
    http.begin(serverURL);

    Serial.println("\n[HTTP] POST...");
    int httpCode = http.sendRequest("POST", frame_buffer->buf, frame_buffer->len);

    if (httpCode > 0) {
      if (httpCode == HTTP_CODE_OK) {
        String response = http.getString();

        if (response.length() > 0) {
          if (response == "OPEN" && servo.read() <= 0) {
            Serial.println("[AUTOMATIC COMMAND] Open the barrier...\n");
            servo.write(90);
            delay(5000);
          } else if ((response == "OPEN" || response == "CLOSE") && servo.read() > 0) {
              Serial.println("[AUTOMATIC COMMAND] Close the barrier...\n");
              servo.write(0);
          } else if (response == "CLOSE" && servo.read() <= 0) {
              buttonState = digitalRead(BUTTON_PIN_NUM);

              if (buttonState == HIGH) {
                Serial.println("[MANUAL COMMAND] Open the barrier...\n");
                servo.write(90);
                delay(5000);
              }
          }
        }
      }
    } else {
      Serial.printf("[HTTP] POST failed with error: %s!\n", http.errorToString(httpCode).c_str());
    }
  } else {
    Serial.println("An error occurred while trying to connect to the network!\n");
  }

  http.end();
  esp_camera_fb_return(frame_buffer);
  delay(3000);
}
