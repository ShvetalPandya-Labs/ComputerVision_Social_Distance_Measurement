import numpy as np
import cv2 as cv

from imutils.video import FPS
import imutils


class SlowVideoStream:
    def __init__(self, path):
        # Open a pointer to the video stream and start the FPS timer
        self.stream = cv.VideoCapture(path)
        self.fps = FPS().start()
        self.frame = np.asarray([])
        self.grabbed = True

    def video_stream_read(self, resize=True, width=700, rgb_convert=False, display_text=False):
        # grab the frame from the threaded video file stream
        (self.grabbed, self.frame) = self.stream.read()
        if not self.grabbed:
            return
        if resize:
            # resize the frame and convert it to grayscale (while still retaining three channels)
            self.frame = imutils.resize(self.frame, width)

        if rgb_convert:
            self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
            self.frame = np.dstack([self.frame, self.frame, self.frame])

        if display_text:
            # display a piece of text to the frame (so we can benchmark fairly against the fast method)
            cv.putText(self.frame, "Slow Method", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return self.frame
