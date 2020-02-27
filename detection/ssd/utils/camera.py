import threading

import cv2


def open_cam_rtsp(uri):
    '''
    :param uri:
    :return:
    '''
    return cv2.VideoCapture(uri)


def grab_img(cam):
    """This 'grab_img' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab a new image and put it
    into the global 'img_handle', until 'thread_running' is set to False.
    """

    while cam.thread_running:
        ret, cam.img_handle = cam.cap.read()
        print(cam.thread_running)
        if ret == False:
            cam.release()
            cam.stop()
            break


class Camera:
    """Camera class which supports reading images from theses video sources:

    1. Video file
    2. Image (jpg, png, etc.) file, repeating indefinitely
    3. RTSP (IP CAM)
    4. USB webcam
    5. Jetson onboard camera
    """

    def __init__(self, rtsp_url):

        self.is_opened = False
        self.use_thread = False
        self.thread_running = False
        self.img_handle = None
        self.img_width = 0
        self.img_height = 0
        self.cap = None
        self.fps = 1000
        self.thread = None
        self.rtsp_url = rtsp_url

    def open(self):
        """Open camera based on command line arguments."""
        assert self.cap is None, 'Camera is already opened!'

        self.use_thread = True

        self.cap = open_cam_rtsp(
            self.rtsp_url
        )
        if self.cap.isOpened():
            # Try to grab the 1st image and determine width and height
            _, img = self.cap.read()
            if img is not None:
                self.img_height, self.img_width, _ = img.shape
                self.is_opened = True

    def start(self):
        assert not self.thread_running
        if self.use_thread:
            self.thread_running = True
            self.thread = threading.Thread(target=grab_img, args=(self,))
            self.thread.start()

    def stop(self):
        self.thread_running = False
        if self.use_thread:
            self.thread.join()

        self.is_opened = False

    def read(self):
        return self.img_handle

    def release(self):
        assert not self.thread_running
        if self.cap != 'OK':
            self.cap.release()
