import asyncio
from gpiod.line import Direction, Value
import gpiod
import time


async def buzzer():
    # Configurar o chip e a linha GPIO
    CHIP = "/dev/gpiochip4"  # Verifique qual gpiochip corresponde ao seu pino
    LINE_OFFSET = 1  # Altere para o número correto do GPIO

    with gpiod.request_lines(
        CHIP,
        consumer="blink-example",
        config={
            LINE_OFFSET: gpiod.LineSettings(
                direction=Direction.OUTPUT, output_value=Value.INACTIVE
            )
        },
    ) as request:
        request.set_value(LINE_OFFSET, Value.ACTIVE)
        await asyncio.sleep(1)
        request.set_value(LINE_OFFSET, Value.INACTIVE)
