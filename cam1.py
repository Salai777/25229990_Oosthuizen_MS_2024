from picamera2 import Picamera2, Preview
from libcamera import Transform
import time
import cv2


###############################################################################################################################


###############################################################################################################################

picam2 = Picamera2()
video_config = picam2.create_video_configuration(
    main={
        "size": (640, 480),
        "format": "BGR888"  # Choose your desired color space format (e.g., YUV420, RGB888, etc.)
    },
    controls={"FrameDurationLimits": (33000, 33000)}
)
picam2.configure(video_config)

watch = time.time()
#start = time.time()
last_frame_time = 0
new_frame_time = 0
fps = 0
FPS = 0

cv2.namedWindow('WINDOW')


try:
    picam2.start()

    i = 0
    while(True):
        i = i + 1

        frame = picam2.capture_array()  # Capture a frame as a numpy array
        last_frame_time = new_frame_time
        new_frame_time = time.time()

        if (i >= 2):
            fps = fps + 1/(new_frame_time - last_frame_time)
            FPS = fps/(i-1)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('WINDOW', frame_bgr)  # Display the frame

        if((time.time() - watch)>= 1.0): # ONCE A SECOND
            print( str(FPS) + " frames per second")
            print("Array shape:", frame.shape)
            watch = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    picam2.close()
    cv2.destroyAllWindows()
