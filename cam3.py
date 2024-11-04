from picamera2 import Picamera2, Preview
from libcamera import Transform
import time
import cv2
import numpy as np


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

###############################################################################################################################

fps = 35.0
fps_parameter = int(1000000/fps)
fps = float(1000000/fps_parameter)

model = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
facecascade = cv2.CascadeClassifier(model)

picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (640, 480)}, 
                                                 controls={"FrameDurationLimits": (fps_parameter, fps_parameter)}) # configure camera to lower resolution
picam2.set_controls({"NoiseReductionMode": "HighQuality" })  # reduce noise for dark conditions

picam2.configure(video_config)

cv2.namedWindow('WINDOW')
cv2.setMouseCallback('WINDOW', mouseRGB)

chroma_similarity = 100
skin_chroma = np.zeros(2, dtype=np.float32) # instantiate skin chroma
is_chroma_selected = False

print(str(fps) + "frames per second")
watch = time.time()
try:
    picam2.start()

    seconds = 0
    i = 0
    while(True):
        i = i + 1

        frame_rgb = picam2.capture_array()  # Capture a frame as a NUMPY array

        frame_grey = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        frame_yuv = np.array(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YUV), dtype=np.float32)


        #face = facecascade.detectMultiScale(frame_grey, 1.1, 6)

        #for (x, y, w, h,) in face:

            #cv2.rectangle(frame_bgr, (x,y), (x+w, y+h), (0, 0, 255)) # Normal rectangle

        if(is_chroma_selected):
            cv2.imshow('WINDOW', chroma_key_display(frame_yuv, skin_chroma))  # Display the frame
        else:
            cv2.imshow('WINDOW', frame_bgr)  # Display the frame

        if((time.time() - watch)>= 1.0): # PRINT ONCE A SECOND
            print("Array shape:", frame_rgb.shape)
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
