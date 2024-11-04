from picamera2 import Picamera2, Preview
from libcamera import Transform
import time
import cv2


###############################################################################################################################


###############################################################################################################################

fps = 35
fps_parameter = int(1000000/fps)
fps = float(1000000/fps_parameter)

model = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
facecascade = cv2.CascadeClassifier(model)

picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (640, 480)}, controls={"FrameDurationLimits": (fps_parameter, fps_parameter)}) # configure camera to lower resolution
picam2.configure(video_config)

cv2.namedWindow('WINDOW')

print(str(fps) + "frames per second")
watch = time.time()
try:
    picam2.start()

    i = 0
    while(True):
        i = i + 1

        frame_rgb = picam2.capture_array()  # Capture a frame as a numpy array

        frame_grey = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        face = facecascade.detectMultiScale(frame_grey, 1.1, 6)

        for (x, y, w, h,) in face:

            cv2.rectangle(frame_bgr, (x,y), (x+w, y+h), (0, 0, 255)) # Normal rectangle

        cv2.imshow('WINDOW', frame_bgr)  # Display the frame

        if((time.time() - watch)>= 1.0): # ONCE A SECOND
            print("Array shape:", frame_rgb.shape)
            watch = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    picam2.close()
    cv2.destroyAllWindows()
