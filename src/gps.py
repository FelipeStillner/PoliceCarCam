import serial
import main as g


def parse_gpgll_message(line):
    """
    Função para extrair latitude e longitude de uma mensagem NMEA do tipo $GPGLL.
    """
    try:
        parts = line.split(",")
        if len(parts) >= 6 and parts[0] == "$GPGLL":
            # Latitude
            raw_latitude = parts[1]
            lat_direction = parts[2]
            # Longitude
            raw_longitude = parts[3]
            lon_direction = parts[4]

            # Converter latitude e longitude para formato decimal
            latitude = convert_to_decimal(raw_latitude, lat_direction)
            longitude = convert_to_decimal(raw_longitude, lon_direction)

            return latitude, longitude
    except Exception as e:
        print(f"Erro ao parsear a mensagem: {e}")
    return None, None


def convert_to_decimal(raw_value, direction):
    """
    Converte coordenadas no formato NMEA (graus e minutos) para decimal.
    """
    if not raw_value or not direction:
        return None

    # Separar graus e minutos
    degrees = int(raw_value[:2]) if direction in ["N", "S"] else int(raw_value[:3])
    minutes = float(raw_value[2:]) if direction in ["N", "S"] else float(raw_value[3:])

    # Calcular valor decimal
    decimal = degrees + (minutes / 60)

    # Ajustar sinal para hemisfério sul (S) ou oeste (W)
    if direction in ["S", "W"]:
        decimal = -decimal

    return decimal


def read_gps_from_uart6():
    try:
        # Configurar a porta UART6
        UART_PORT = "/dev/ttyS6"
        BAUD_RATE = 9600  # Taxa de comunicação padrão do módulo GPS (verifique no manual do módulo)
        read_flag = 0

        # Abrir a porta serial
        with serial.Serial(UART_PORT, BAUD_RATE, timeout=1) as ser:
            print(
                f"Lendo dados do GPS na porta {UART_PORT} com baud rate {BAUD_RATE}...\n"
            )

            while read_flag == 0:
                # Ler uma linha de dados do GPS
                line = ser.readline().decode("utf-8", errors="ignore").strip()

                if line:
                    # Verificar mensagens NMEA do tipo $GPGLL
                    if line.startswith("$GPGLL"):
                        latitude, longitude = parse_gpgll_message(line)
                        if latitude is not None and longitude is not None:
                            g.global_latitude = latitude
                            g.global_longitude = longitude
                            print(
                                f"Latitude: {latitude:.6f}, Longitude: {longitude:.6f}"
                            )
                            read_flag = 1
                        else:
                            print("Mensagem GPGLL inválida ou incompleta.")
    except serial.SerialException as e:
        print(f"Erro ao acessar a porta serial: {e}")
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário. Encerrando o script.")
