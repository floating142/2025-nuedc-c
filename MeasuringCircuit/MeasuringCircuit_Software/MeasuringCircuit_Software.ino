#include "SSD1306Wire.h"
//oled 
const int I2C_ADDR = 0x3c;              // olediic地址
#define SDA_PIN 4                       // io4 
#define SCL_PIN 5                       // io5

#include <Wire.h> 
#include <INA226_WE.h> 
#define I2C_ADDRESS 0x40 // 定义 INA226 的 I2C 位址

/* 新建oled屏幕对象，输入IIC地址，SDA和SCL引脚号 */
SSD1306Wire oled(I2C_ADDR, SDA_PIN, SCL_PIN);

INA226_WE ina226 = INA226_WE(I2C_ADDRESS); // 新建INA226对象，设置iic地址

float PowerMax;
//测试屏幕显示
void drawRect(void) {
  for (int16_t i=0; i<oled.getHeight()/2; i+=2) {
    oled.drawRect(i, i, oled.getWidth()-2*i, oled.getHeight()-2*i);
    oled.display();
    delay(50);
  }
}
void setup() {
  Serial.begin(115200); //串口频率
  
 /*  oled初始化 */
  oled.init();
  oled.flipScreenVertically();          
  oled.setContrast(255);                
  drawRect();                           
  oled.clear(); oled.display();        

  Wire.begin(4,5,10000000); 
  ina226.init(); 

  ina226.setResistorRange(0.01, 7.0); 
  ina226.setCorrectionFactor(0.93); 

  Serial.println("INA226OK");

  ina226.waitUntilConversionCompleted();
}
void loop(){
  float shuntVoltage_mV = 0.0; 
  float loadVoltage_V = 0.0;   
  float busVoltage_V = 0.0;    
  float current_mA = 0.0;      
  float power_mW = 0.0;        

  ina226.readAndClearFlags(); 
  shuntVoltage_mV = ina226.getShuntVoltage_mV(); 
  busVoltage_V = ina226.getBusVoltage_V();       
  current_mA = ina226.getCurrent_mA();           
  power_mW = ina226.getBusPower();               
  loadVoltage_V  = busVoltage_V + (shuntVoltage_mV / 1000); 
  if(power_mW>PowerMax) {PowerMax=power_mW;}
  /* 显示字母 */
  oled.setFont(ArialMT_Plain_10);       
  oled.drawString(0,0, "Volt:"+String(busVoltage_V)+"V"); 
  oled.drawString(0,15, "Current:"+String(current_mA)+"mA");
  oled.drawString(0,30, "Power:"+String(power_mW)+"mW");
  oled.drawString(0,45, "PMax:"+String(PowerMax)+"mW");
  oled.display();                      
  Serial.println("V:"+String(busVoltage_V)+"mV"+"I:"+String(current_mA)+"mA"+"P:"+String(power_mW)+"mW");
  delay(1000);
  oled.clear(); 
  oled.display();         
}