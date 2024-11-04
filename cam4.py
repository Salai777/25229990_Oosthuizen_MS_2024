from picamera2 import Picamera2, Preview
from libcamera import ColorSpace
import time
import cv2
import numpy as np
from scipy.signal import butter, filtfilt


###############################################################################################################################

def mouseRGB(event, x, y, flags, params):
    global skin_chroma, is_chroma_selected, frame_yuv
    if event == cv2.EVENT_LBUTTONDOWN: #changes chroma reference
        yuv_frame = cv2.cvtColor(np.array([[frame_bgr[y, x]]]), cv2.COLOR_BGR2YUV)
        skin_chroma = yuv_frame[0, 0, 1:3].astype(np.float32)
        #skin_chroma = np.array([124.0, 164.0], dtype=np.float32)
        print('RGB = ', frame_yuv[y, x], 'chroma = ', skin_chroma)
        is_chroma_selected = True

    if event == cv2.EVENT_RBUTTONDOWN: #shows reference on different frame 
        print("nothing")

def chroma_key(frame, chroma): #returns frame variable of boolean values
    key = frame[:, :, 1:3] - chroma
    key = np.less(np.sum(np.square(key), axis=2), chroma_similarity)
    return key


def chroma_key_display(frame, chroma): #returns frame variable of black and white

    key = chroma_key(frame, chroma)
    return (key * 255).astype(np.uint8)

def searchfaces(num_faces):

    global face_detected, detection_counter, no_detection_counter

    if(num_faces > 0):
        detection_counter = detection_counter + 1
        no_detection_counter = 0
    else:
        no_detection_counter = no_detection_counter + 1
        detection_counter = 0

    if(detection_counter >= 10):
        face_detected = True
    elif(no_detection_counter >= 5):
        face_detected = False


def sample_RGB():

    if len(face) == 1 and face_detected:
        for (x, y, w, h) in face:
            last_x = x
            last_y = y
            last_w = w
            last_h = h
            avs_RGB.append(calc_mean(frame_np, skin_key, x, y, w, h))
    elif len(face) >= 1 and face_detected:
        avs_RGB.append(calc_mean(frame_np, skin_key, last_x, last_y, last_w, last_h))
    elif len(face) == 0 and not face_detected:  # before any detection
        avs_RGB.append([0, 0, 0])
    elif len(face) == 1 and not face_detected:  # first detection
        face_detected = True
        for (x, y, w, h) in face:
            last_x = x
            last_y = y
            last_w = w
            last_h = h
            avs_RGB.append(calc_mean(frame_np, skin_key, x, y, w, h))

def calc_mean():
    print("here")

def findchroma(face):
    global is_chroma_selected




###############################################################################################################################

fps = 35.0
fps_parameter = int(1000000/fps)
fps = float(1000000/fps_parameter)

model = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
facecascade = cv2.CascadeClassifier(model)

picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"},
                                                 controls={"FrameDurationLimits": (fps_parameter, fps_parameter)}) # configure camera to lower resolution
picam2.set_controls({"NoiseReductionMode": "HighQuality" })  # Set to high quality for low ambient lighting 



picam2.configure(video_config)

cv2.namedWindow('WINDOW')
cv2.setMouseCallback('WINDOW', mouseRGB)

chroma_similarity = 100
skin_chroma = np.zeros(2, dtype=np.float32) # instantiate skin chroma
is_chroma_selected = False

face_detected = False
detection_counter = 0
no_detection_counter = 0
last_x = 0
last_y = 0
last_w = 0
last_h = 0
avs_RGB = []

print(str(fps) + "frames per second")
watch = time.time()
try:
    picam2.start()

    seconds = 0
    i = 0
    while(True):
        i = i + 1

        frame_bgr = picam2.capture_array()  # Capture a frame

        frame_grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        frame_yuv = np.array(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV), dtype=np.float32)

        face = facecascade.detectMultiScale(frame_grey, 1.1, 6)

        #for (x, y, w, h,) in face:

            #cv2.rectangle(frame_bgr, (x,y), (x+w, y+h), (0, 0, 255)) # Normal rectangle

        #searchfaces(len(face)) #controls flags
        #sample_RGB(face)       #samples rgb values
        """
        if(not is_chroma_selected):
            if(len(face) > 0):
                findchroma() """

        if(is_chroma_selected):
            cv2.imshow('WINDOW', chroma_key_display(frame_yuv, skin_chroma))  # Display the frame
        else:
            cv2.imshow('WINDOW', frame_bgr)  # Display the frame

        if((time.time() - watch)>= 1.0): # PRINT ONCE A SECOND
            print("Array shape:", frame_bgr.shape)
            seconds = seconds + 1
            print(str(seconds) + "seconds")
            watch = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    picam2.close()
    cv2.destroyAllWindows()
