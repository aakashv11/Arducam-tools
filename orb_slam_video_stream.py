import cv2
import lcm
import time
#import matplotlib.pyplot as plt
from lcmtypes import mbot_video_stream_t
import numpy as np


def stream_handler(channel, data):
    print("ENTERED HANDLER")
    msg = mbot_video_stream_t.decode(data)
    image = cv2.imdecode(np.array(msg.image), cv2.IMREAD_GRAYSCALE)
    image = cv2.flip(image, -1)
    cv2.imwrite("./stream_data/"+str(msg.timestamp)+".png", image)


i = 0
lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
subscription = lc.subscribe("MBOT_VIDEO_STREAM", stream_handler)

try:
    while True:
        lc.handle_timeout(10)
except KeyboardInterrupt:
    cv2.destroyAllWindows()
    pass

