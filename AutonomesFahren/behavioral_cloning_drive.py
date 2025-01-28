import cv2
from SimulatorCommunication import SimulatorCommunication
import behavioral_cloning_utils as utils
import numpy as np
import keras
import os


def start_autonomous_driving(model, stop_thread):
    sim_com = SimulatorCommunication(stop_thread)
    speed_limit = 30
    model = keras.models.load_model(os.path.join(r'models/', model), safe_mode=False)

    while True:
        if not stop_thread():
            cv2.destroyAllWindows()
            break

        try:
            image, speed = sim_com.receive_telemetry()
        except:
            print('com failed')

        if image is not None:
            if speed < speed_limit:
                throttle = 0.8
            else:
                throttle = -0.2

            image2 = utils.preprocess(image)  # apply preprocessing
            image2 = np.array([image2])  # model expects batch of images (4D array)
            steering = model.predict(image2, batch_size=1, verbose=0)[0][0]

            try:
                sim_com.send_controls(steering, throttle)
            except:
                print('socket closed')

            cv2.imshow("Car POV", image)
            cv2.waitKey(1)


def main():
    start_autonomous_driving('test')


if __name__ == '__main__':
    main()
