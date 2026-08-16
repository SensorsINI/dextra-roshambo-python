from consumer import brightness_controller
import time

b=brightness_controller()

tstart=time.time()
while True:
    b.set_screen_brightness(.01)
    time.sleep(5)
    print(f'elapsed {time.time()-tstart:.0f}s')
    b.set_screen_brightness(1)
    time.sleep(5)