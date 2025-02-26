from RPLCD.i2c import CharLCD

# Configuração do LCD
LCD = CharLCD(i2c_expander="PCF8574", address=0x27, port=8, cols=16, rows=2, dotsize=8)


def escrever_lcd(texto_linha1, texto_linha2):
    print(texto_linha1, texto_linha2)
    LCD.clear()
    LCD.write_string(texto_linha1)
    LCD.crlf()  # Pula para a segunda linha
    LCD.write_string(texto_linha2)
