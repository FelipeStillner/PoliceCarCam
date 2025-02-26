from gps import read_gps_from_uart6
import gpiod
import select
import asyncio
from datetime import timedelta
from gpiod.line import Bias, Edge


async def async_watch_line_value(stop_event, event_listener):
    CHIP_PATH = "/dev/gpiochip0"
    LINE_OFFSET = 14
    # Configura o GPIO com debounce e detecção de bordas
    with gpiod.request_lines(
        CHIP_PATH,
        consumer="async-watch-line-value",
        config={
            LINE_OFFSET: gpiod.LineSettings(
                edge_detection=Edge.FALLING,
                bias=Bias.PULL_UP,
                debounce_period=timedelta(milliseconds=10),
            )
        },
    ) as request:
        loop = asyncio.get_event_loop()
        poll = select.poll()
        poll.register(request.fd, select.POLLIN)

        while not stop_event.is_set():
            # Aguarda eventos do GPIO ou do stop_event
            events = await loop.run_in_executor(
                None, poll.poll, 1000
            )  # Timeout de 1000ms
            for fd, _event in events:
                if stop_event.is_set():
                    return
                if fd == request.fd:
                    # Lê e processa eventos de borda
                    for event in request.read_edge_events():
                        event_listener.set()
                        read_gps_from_uart6()
