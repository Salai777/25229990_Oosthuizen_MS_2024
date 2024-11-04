from picamera2 import Picamera2, Preview
from libcamera import Transform
import time
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

###############################################################################################################################

def mouseRGB(event, x, y, flags, params):
    global skin_chroma, is_chroma_selected, frame_yuv_32, chroma_similarity
    if event == cv2.EVENT_LBUTTONDOWN: #changes chroma reference
        #yuv_frame = cv2.cvtColor(np.array([[frame_bgr[y, x]]]), cv2.COLOR_BGR2YUV)
        #skin_chroma = yuv_frame[0, 0, 1:3].astype(np.float32)
        #skin_chroma = np.array([124.0, 164.0], dtype=np.float32)
        #print('RGB = ', frame_yuv_32[y, x], 'chroma = ', skin_chroma)
        chroma_similarity = chroma_similarity - 5
        #is_chroma_selected = True
        print("nothing")

    if event == cv2.EVENT_RBUTTONDOWN: #shows reference on different frame 
        print("nothing")
        #chroma_similarity = chroma_similarity - 5

def chroma_key(frame, chroma): #returns frame variable of boolean values
    key = frame[:, :, 1:3] - chroma
    key = np.less(np.sum(np.square(key), axis=2), chroma_similarity)
    return key


def chroma_key_display(frame, chroma): #returns frame variable of black and white

    key = chroma_key(frame, chroma)
    return (key * 255).astype(np.uint8)

def forehead(frame_bgr, x, y, w, h):
    
    x_distance = int(w*0.45) #increase to make smaller
    y_distance = int(h*0.85) #increase to make smaller
    offset = int(h*0.33) #increase to move up

    x_start = x + (x_distance // 2)
    y_start = y + (y_distance // 2) - offset
    x_end = x + w - (x_distance // 2)
    y_end = y + h - (y_distance // 2) - offset

    # Ensure the adjusted rectangle is within image bounds
    x_start = max(x_start, 0)
    y_start = max(y_start, 0)
    x_end = min(x_end, frame_bgr.shape[1])
    y_end = min(y_end, frame_bgr.shape[0])

    cv2.rectangle(frame_bgr, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1) # Forehead rectangle
                
def cheeks(frame_bgr, x, y, w, h):

    x_distance = int(w*0.84) #increase to make smaller
    y_distance = int(h*0.8) #increase to make smaller
    x_offset = int(w*0.26) #increase to apart
    y_offset = int(h*-0.15) #increase to move up

    x_start = x + (x_distance // 2) - x_offset
    y_start = y + (y_distance // 2) - y_offset
    x_end = x + w - (x_distance // 2) - x_offset
    y_end = y + h - (y_distance // 2) - y_offset

    # Ensure the adjusted rectangle is within image bounds
    x_start = max(x_start, 0)
    y_start = max(y_start, 0)
    x_end = min(x_end, frame_bgr.shape[1])
    y_end = min(y_end, frame_bgr.shape[0])

    cv2.rectangle(frame_bgr, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1) # left cheek rectangle

    x_start = x + (x_distance // 2) + x_offset
    x_end = x + w - (x_distance // 2) + x_offset
    # Ensure the adjusted rectangle is within image bounds
    x_start = max(x_start, 0)
    x_end = min(x_end, frame_bgr.shape[1])

    cv2.rectangle(frame_bgr, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1) # right cheek rectangle

def find_chroma(x, y, w, h):
    global frame_yuv_32

    x_distance = int(w*0.45) #increase to make smaller
    y_distance = int(h*0.85) #increase to make smaller
    offset = int(h*0.33) #increase to move up

    x_start = x + (x_distance // 2)
    y_start = y + (y_distance // 2) - offset
    x_end = x + w - (x_distance // 2)
    y_end = y + h - (y_distance // 2) - offset

    # Ensure the adjusted rectangle is within image bounds
    x_start = max(x_start, 0)
    y_start = max(y_start, 0)
    x_end = min(x_end, frame_bgr.shape[1])
    y_end = min(y_end, frame_bgr.shape[0])

    U_mean = np.mean(frame_yuv_32[y_start:y_end, x_start:x_end, 1])
    V_mean = np.mean(frame_yuv_32[y_start:y_end, x_start:x_end, 2])

    new_skin_chroma = np.array([U_mean, V_mean], dtype=np.float32)

    return new_skin_chroma

def searchfaces(num_faces):

    global face_detected, detection_counter, no_detection_counter, new_face, is_chroma_selected, avs_RGB

    if(num_faces > 0 and face_detected):
        no_detection_counter = 0
    elif(num_faces > 0 and not face_detected):
        detection_counter += 1
    elif(num_faces == 0 and face_detected):
        no_detection_counter += 1
    elif(num_faces == 0 and not face_detected):
        detection_counter = 0

    if(detection_counter >= 20 and not face_detected):
        face_detected = True #true when face is in frame
        new_face = True      #true only at detection
        detection_counter = 0
    elif(no_detection_counter >= 20 and face_detected):
        face_detected = False
        is_chroma_selected = False
        no_detection_counter = 0
        avs_RGB.clear()

def setchroma(face):

    global is_chroma_selected, skin_chroma

    if((len(face) == 1)): #set chroma when face detetced

        for (x, y, w, h,) in face:

            if(not is_chroma_selected and face_detected):
                skin_chroma =  find_chroma(x, y, w, h)
                print("CHROMA SELECTED")
                is_chroma_selected = True
                new_face = False

                #cv2.rectangle(frame_bgr, (x,y), (x+w, y+h), (0, 0, 255)) # Normal rectangle
                #forehead(frame_bgr, x, y, w, h)
                #cheeks(frame_bgr, x, y, w, h)

def get_RGB(face):
    global avs_RGB, last_x, last_y, last_w, last_h

    if (len(face) == 1): #length will be 1 first frame that chroma is selected
        for (x, y, w, h) in face:
            last_x = x
            last_y = y
            last_w = w
            last_h = h
            avs_RGB.append(ROI_mean(x, y, w, h))
    elif (len(face) > 1 or len(face) == 0):
        avs_RGB.append(ROI_mean(last_x, last_y, last_w, last_h))

    if(len(avs_RGB) > signal_length):
        del avs_RGB[0]

def ROI_mean(x, y, w, h): #find the mean over the ROI within the facial box and chroma similarity
    global skin_chroma, frame_yuv_32, frame_rgb_32

    ROI_rgb = frame_rgb_32[y:y+h, x:x+w,:]
    ROI_yuv = frame_yuv_32[y:y+h, x:x+w,:]
    frame_bool = chroma_key(ROI_yuv, skin_chroma)
    avs = np.mean(ROI_rgb[frame_bool], axis = 0)

    return avs

def get_BPM():
    global avs_RGB, fps

    RED = [avs[0]*0.7682 for avs in avs_RGB] 
    GREEN = [avs[1]*0.5121 for avs in avs_RGB] 
    BLUE = [avs[2]*0.3841 for avs in avs_RGB] 

    R_n = []
    G_n = [] #normalised colour channels
    B_n = []

    start = int(1.5*fps)
    stop = int(signal_length - start)
    mean_length = 1.5*fps
    for z in range(start, stop): #normalising 
        slice_start = int(z - mean_length/2)
        slice_stop = int(z + mean_length/2)
        R_n.append(RED[z]/np.mean(RED[slice_start:slice_stop]))
        G_n.append(GREEN[z]/np.mean(GREEN[slice_start:slice_stop]))
        B_n.append(BLUE[z]/np.mean(BLUE[slice_start:slice_stop]))

    X_n = [3*Rn - 2*Gn for Rn, Gn in zip(R_n, G_n)]
    Y_n = [1.5*Rn + Gn - 1.5*Bn for Rn, Gn, Bn in zip(R_n, G_n, B_n)]

    #red = butter_bandpass_filter(RED, lowcut, highcut, fps)
    #green = butter_bandpass_filter(GREEN, lowcut, highcut, fps)
    #blue = butter_bandpass_filter(BLUE, lowcut, highcut, fps)

    Xf = butter_bandpass_filter(X_n, lowcut, highcut, fps)
    Yf = butter_bandpass_filter(Y_n, lowcut, highcut, fps)

    alpha = np.std(Xf)/np.std(Yf)

    HR = [xf - alpha*yf for xf, yf in zip(Xf, Yf)]

    print("length of Xf is " + str(len(Xf)))

    axis = np.arange(len(RED))
    x = axis/fps #time axis
    plt.figure(figsize=(10, 6))

    plt.plot(x, RED, color='blue', marker='o')
    #plt.plot(x, green, label='Green Channel', color='green', marker='o')
    #plt.plot(x, blue, label='Blue Channel', color='blue', marker='o')
    # Adding titles and labels
    plt.title('Heart Rate signal using the Chrominance Method')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()          # Show legend
    plt.grid()
    plt.show()



def looptimes():
    global i, running_average, new_loop_time, last_loop_time, LOOPS_PS

    last_loop_time = new_loop_time
    new_loop_time = time.time()

    if (i >= 2):

        running_average = running_average + 1/(new_loop_time - last_loop_time)
        LOOPS_PS = running_average/(i-1)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # Get filter coefficients for a Butterworth bandpass filter
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band', analog=False)
    # Apply the filter to the data
    filtered_data = filtfilt(b, a, data)
    return filtered_data

lowcut = 0.7  # Low cut-off frequency
highcut = 3.0  # High cut-off frequency

###############################################################################################################################

fps = 30.0
fps_parameter = int(1000000/fps)
fps = float(1000000/fps_parameter)
signal_length = int(8*fps) # length for 8 seconds

model = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
facecascade = cv2.CascadeClassifier(model)

picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (320, 240), "format": "BGR888"},
                                                 controls={"FrameDurationLimits": (fps_parameter, fps_parameter)}) # configure camera to lower resolution
picam2.set_controls({"NoiseReductionMode": "HighQuality" })  # reduce noise for dark conditions

picam2.configure(video_config)

#cv2.namedWindow('WINDOW')
#cv2.setMouseCallback('WINDOW', mouseRGB)

chroma_similarity = 80
skin_chroma = np.zeros(2, dtype=np.float32) # instantiate skin chroma
is_chroma_selected = False

face_detected = False
new_face = False
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
    last_loop_time = 0
    new_loop_time = 0
    running_average = 0
    LOOP_TIME = 0
    while(True):
        i = i + 1

        frame = picam2.capture_array() #capture frame
        looptimes()
        
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_rgb_32 = np.array(frame, dtype=np.float32)  # convert to float NUMPY array
        frame_yuv_32 = np.array(cv2.cvtColor(frame, cv2.COLOR_RGB2YUV), dtype=np.float32)
        
        face = facecascade.detectMultiScale(frame_grey, 1.1, 6)
        
        searchfaces(len(face)) #determine if person is in the frame and set flags

        if(face_detected and new_face):
            setchroma(face) #for first detection (sets chroma and flags)
            get_RGB(face)
        elif(face_detected and is_chroma_selected):
                get_RGB(face) #sample RGB values
        

        if(is_chroma_selected):
            cv2.imshow('WINDOW', chroma_key_display(frame_yuv_32, skin_chroma))  # Display the frame
        else:
            cv2.imshow('WINDOW', frame_bgr)  # Display the frame
        
        
        if((time.time() - watch)>= 1.0): # PRINT ONCE A SECOND
            print( str(1/(new_loop_time - last_loop_time)) + " loops per second")
            if(len(avs_RGB) > 0):
                print("Array shape:", avs_RGB[0].shape)
                print("length is " + str(len(avs_RGB)))
            seconds = seconds + 1
            print(str(seconds) + "seconds\n" )
            #if(face_detected):
                #print("DETECTION")
            #else:
                #print("...")
            watch = time.time()
            #get_BPM() #calculate BPM

        if cv2.waitKey(1) & 0xFF == ord('q'):
            get_BPM()
            break

    

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    picam2.close()
    cv2.destroyAllWindows()
