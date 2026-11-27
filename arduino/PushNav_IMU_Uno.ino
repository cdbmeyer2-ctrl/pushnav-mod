/*
  PushNav IMU bridge - Arduino Uno Rev3 + GY-521 / MPU-6050

  Hardware:
    GY-521 VCC -> 5V ONLY if your specific GY-521 breakout is confirmed
                 to include a 5V regulator and safe I2C pull-ups/level shifting.
                 Otherwise power/level-shift according to its documentation.
    GY-521 GND -> GND
    GY-521 SDA -> A4 (SDA)
    GY-521 SCL -> A5 (SCL)
    AD0 -> GND (I2C address 0x68), unless your board is configured otherwise.

  Serial protocol (little-endian):
    sync[2] = 0xAA, 0x55
    sequence uint16
    micros uint32
    ax, ay, az, gx, gy, gz, temperature int16 each
    crc8 uint8 (CRC-8/ATM, poly 0x07, init 0)
  Total packet size: 23 bytes.

  Sensor configuration:
    accelerometer +/-2 g; gyro +/-250 degrees/sec; DLPF ~44/42 Hz;
    nominal sample rate 100 Hz.
*/
#include <Wire.h>
#include <stdint.h>

static const uint8_t MPU_ADDR = 0x68;
static const uint8_t REG_PWR_MGMT_1 = 0x6B;
static const uint8_t REG_SMPLRT_DIV = 0x19;
static const uint8_t REG_CONFIG = 0x1A;
static const uint8_t REG_GYRO_CONFIG = 0x1B;
static const uint8_t REG_ACCEL_CONFIG = 0x1C;
static const uint8_t REG_ACCEL_XOUT_H = 0x3B;
static const uint8_t REG_WHO_AM_I = 0x75;

static const uint32_t SAMPLE_PERIOD_US = 10000UL; // 100 Hz
static uint16_t sequenceNumber = 0;
static uint32_t nextSampleUs = 0;
static bool sensorReady = false;

struct __attribute__((packed)) ImuPacket {
  uint8_t sync0;
  uint8_t sync1;
  uint16_t sequence;
  uint32_t timestampUs;
  int16_t ax;
  int16_t ay;
  int16_t az;
  int16_t gx;
  int16_t gy;
  int16_t gz;
  int16_t temperature;
  uint8_t crc;
};

static uint8_t crc8(const uint8_t *data, size_t length) {
  uint8_t crc = 0;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
    }
  }
  return crc;
}

static bool writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

static bool readRegisters(uint8_t startReg, uint8_t *buffer, uint8_t count) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(startReg);
  if (Wire.endTransmission(false) != 0) return false;
  uint8_t received = Wire.requestFrom(MPU_ADDR, count, (uint8_t)true);
  if (received != count) return false;
  for (uint8_t i = 0; i < count; ++i) buffer[i] = Wire.read();
  return true;
}

static int16_t readI16(const uint8_t *p) {
  return (int16_t)(((uint16_t)p[0] << 8) | p[1]);
}

static bool initializeMpu() {
  uint8_t who = 0;

  if (!readRegisters(REG_WHO_AM_I, &who, 1)) {
    return false;
  }

  // Standard MPU-6050 identity is 0x68.
  // This particular module consistently reports 0x75.
  // Accept 0x75 only with successful configuration and
  // measurement-register verification below.
  if (who != 0x68 && who != 0x75) {
    return false;
  }

  // Wake device; use gyro X as PLL clock.
  if (!writeRegister(REG_PWR_MGMT_1, 0x01)) return false;
  delay(100);

  // Configure digital low-pass filter and sample rate.
  if (!writeRegister(REG_CONFIG, 0x03)) return false;
  if (!writeRegister(REG_SMPLRT_DIV, 9)) return false;

  // Gyroscope +/-250 degrees/sec.
  if (!writeRegister(REG_GYRO_CONFIG, 0x00)) return false;

  // Accelerometer +/-2g.
  if (!writeRegister(REG_ACCEL_CONFIG, 0x00)) return false;

  // Verify configuration readback.
  uint8_t value;

  if (!readRegisters(REG_PWR_MGMT_1, &value, 1)) return false;
  if ((value & 0x40) != 0) return false;

  if (!readRegisters(REG_CONFIG, &value, 1)) return false;
  if (value != 0x03) return false;

  if (!readRegisters(REG_SMPLRT_DIV, &value, 1)) return false;
  if (value != 9) return false;

  if (!readRegisters(REG_GYRO_CONFIG, &value, 1)) return false;
  if (value != 0x00) return false;

  if (!readRegisters(REG_ACCEL_CONFIG, &value, 1)) return false;
  if (value != 0x00) return false;

  // Verify that all 14 measurement bytes can be read.
  uint8_t testData[14];

  if (!readRegisters(REG_ACCEL_XOUT_H, testData, 14)) {
    return false;
  }

  return true;

}

static void sendPacket() {
  uint8_t raw[14];
  if (!readRegisters(REG_ACCEL_XOUT_H, raw, sizeof(raw))) return;

  ImuPacket packet;
  packet.sync0 = 0xAA;
  packet.sync1 = 0x55;
  packet.sequence = sequenceNumber++;
  packet.timestampUs = micros();
  packet.ax = readI16(&raw[0]);
  packet.ay = readI16(&raw[2]);
  packet.az = readI16(&raw[4]);
  packet.temperature = readI16(&raw[6]);
  packet.gx = readI16(&raw[8]);
  packet.gy = readI16(&raw[10]);
  packet.gz = readI16(&raw[12]);
  packet.crc = crc8((const uint8_t *)&packet, sizeof(packet) - 1);
  Serial.write((const uint8_t *)&packet, sizeof(packet));
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000UL);
  sensorReady = initializeMpu();

  // A short ASCII banner is intentionally avoided: the host expects binary frames.
  // A failed sensor simply results in no packets; host reports a timeout.
  nextSampleUs = micros();
}

void loop() {
  if (!sensorReady) {
    delay(250);
    sensorReady = initializeMpu();
    return;
  }

  uint32_t now = micros();
  if ((int32_t)(now - nextSampleUs) >= 0) {
    nextSampleUs += SAMPLE_PERIOD_US;
    sendPacket();
    // If delayed by more than one period, resynchronize rather than burst packets.
    if ((int32_t)(now - nextSampleUs) > (int32_t)SAMPLE_PERIOD_US) {
      nextSampleUs = now + SAMPLE_PERIOD_US;
    }
  }
}
