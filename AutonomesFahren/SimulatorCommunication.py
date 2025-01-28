import cv2
import numpy as np
import base64
import json
from io import BytesIO
from PIL import Image

from UDPCommunication import UDPCommunication


class SimulatorCommunication:
    """
    Communication between Unity3D Driving Simulator and Python script
    """

    def __init__(self, stop_thread):
        self.sock = UDPCommunication(udp_ip="127.0.0.1",
                                     port_tx=8000,
                                     port_rx=8001,
                                     enable_rx=True,
                                     suppress_warnings=True,
                                     stop_thread=stop_thread)

    def send_controls(self, steering: float, throttle: float):
        """
        Send controls to simulator.
        :param steering: target steering (-1.0: full left | 0.0: straight | 1.0: full right)
        :param throttle: target throttle position (1.0: full throttle | 0.0: idle/roll | -1.0 brake/reverse)
        """

        steering = int(10000 * np.clip(steering, a_min=-1, a_max=1))
        throttle = int(10000 * np.clip(throttle, a_min=-1, a_max=1))
        self.sock.send_data(f"Steering:{steering:6d}|Throttle:{throttle:6d}")

    def receive_telemetry(self):
        """
        Receive telemetry data from simulator if available
        :return: Tuple of size 2:
            POV of car in simulation as BGR image in cv2 format (or None if not available);
            speed in km/h (or 0 if not available)
        """
        msg = self.sock.read_received_data()

        if msg is not None and msg.startswith('{"img"') and msg.endswith('}'):
            data = json.loads(msg)

            pil_image = Image.open(BytesIO(base64.b64decode(data["img"])))
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            return cv_image, int(data["speed"])

        return None, 0
