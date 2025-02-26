import asyncio


# Função assíncrona para verificar conexão Wi-Fi
async def is_wifi_connected(wifi_event):
    while True:
        try:
            # Executa o comando nmcli para verificar conexões ativas
            process = await asyncio.create_subprocess_exec(
                "nmcli",
                "-t",
                "-f",
                "STATE,TYPE",
                "connection",
                "show",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                print(f"Erro no comando nmcli: {stderr.decode()}")
                wifi_event.clear()
                continue

            for line in stdout.decode().splitlines():
                active, conn_type = line.split(":")
                if active == "activated" and conn_type == "802-11-wireless":
                    print("Conectado ao Wi-Fi!")
                    wifi_event.set()
                    break
            else:
                print("Não há conexão Wi-Fi ativa.")
                wifi_event.clear()
        except Exception as e:
            print(f"Erro inesperado: {e}")
            wifi_event.clear()

        await asyncio.sleep(5)
