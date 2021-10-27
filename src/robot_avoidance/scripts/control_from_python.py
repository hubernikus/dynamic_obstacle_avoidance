import time
from control import FrankaZMQInterface
import os.path

if __name__ == "__main__":
    # create interface with default state_uri and command_uri
    interface = FrankaZMQInterface()

    desired_frequency = 500
    start = time.time()
    k = 0
    while interface.is_connected():
        now = time.time()
        state = interface.get_robot_state()
        print(state)

        elapsed = time.time() - now
        sleep_time = (1.0 / desired_frequency) - elapsed
        if sleep_time > 0.0:
            time.sleep(sleep_time)
        k = k + 1

        print("Average rate: ", k / (time.time() - start))
