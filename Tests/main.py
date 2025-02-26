import asyncio
import time
import backupDataBase 
import gpioRisingDetec as gpio
import gpsCapture
import wifiConnect as wc
import camera
import lcd
import buzzer

async def main():
    #Configuração do banco
    host = "0.tcp.sa.ngrok.io:15119"
    database = "postgres"
    user = "postgres"
    password = "postgres"
    query = "SELECT * FROM public.vehicle"
    output_file = "backup.csv"

    #Configuração gpio
    chip_path = "/dev/gpiochip1" 
    line_offset = 31
    stop_event = asyncio.Event()

    #Parametro wifi
    wifi_event = asyncio.Event()

    buttom_event = asyncio.Event()
    video_output = "video.mp4"

    lcd.escrever_lcd("Polce Car Cam", "")
    buzzer.buzzer()

    try:
        await asyncio.gather(
            wc.is_wifi_connected(wifi_event),
            backupDataBase.backup_if_wifi(host, database, user, password, query, output_file, wifi_event),
            gpio.async_watch_line_value(chip_path, line_offset, stop_event, gpsCapture.read_gps_from_uart6, buttom_event),
            camera.record_video_with_location(buttom_event),
            camera.capture_frame()
        )
    except KeyboardInterrupt:
        print("\nInterrupção manual detectada. Encerrando...")
    finally:
        stop_event.set()  # Sinaliza para parar a tarefa
        print("Monitoramento encerrado.")

if __name__ == "__main__":
    asyncio.run(main())
