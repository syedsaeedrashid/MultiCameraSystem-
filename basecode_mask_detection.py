import cv2
import threading
import numpy as np
import glob
import random

#creating a class of camera to create camera objects in separate threads
class camThread(threading.Thread):
    def __init__(self, Name, IPAddress, recording, YOLO):
        threading.Thread.__init__(self)
        self.Name = Name
        self.IPAddress = IPAddress
        self.recording = recording
        self.YOLO = YOLO
    def run(self):
        print("Starting " + self.Name)
        camProcess(self.Name, self.IPAddress, self.recording, self.YOLO)


def camProcess(Name, IPAddress, recording, YOLO):
    if recording==0 and YOLO==0:   #if recording mode is set off, then only stream, without YOLO Detection
        cv2.namedWindow(Name)
        cam = cv2.VideoCapture(IPAddress)
        if cam.isOpened():
            ret, frame = cam.read()
        else:
            ret = False

        while ret:
            cv2.imshow(Name, frame)
            ret, frame = cam.read()
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
        cam.release()
        cv2.destroyWindow(Name)
        
    elif recording==1 and YOLO==0:				#if recording mode is set on, then record the stream, without YOLO Detection
        cv2.namedWindow(Name)
        cam = cv2.VideoCapture(IPAddress)
        width = int(cam.get(3))
        height = int(cam.get(4))
        size = (width, height)
        
        # frame of size is being created and stored in .avi file
        outputFile = cv2.VideoWriter(
            Name+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
        
        if cam.isOpened():
            ret, frame = cam.read()
        else:
            ret = False

        while ret:
            cv2.imshow(Name, frame)
            ret, frame = cam.read()
            # saves the frame from camera 
            outputFile.write(frame)
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
        cam.release()
        outputFile.release()
        cv2.destroyWindow(Name)
        
    elif recording==0 and YOLO==1:
        cv2.namedWindow(Name)
        # read pretrained network and its weights
        yolo_net = cv2.dnn.readNet("yolov3_face_mask.weights", "mask-yolov3 - Copy.cfg")
        classes = ["no_mask", "mask"]
        names_of_layers = yolo_net.getLayerNames()
        out_layers = [names_of_layers[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3)) #randomly assigning x separate colors for x classes 

        font = cv2.FONT_HERSHEY_SIMPLEX     #Simplex font is usually the most thin, hence used this
        cam = cv2.VideoCapture(IPAddress)
        
        while True:
            ret, frame = cam.read()
            frame = cv2.resize(frame, None, fx=1, fy=1) 
            height, width, channels = frame.shape

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False) #Blob Detection to identify regions
            #blob = cv2.dnn.blobFromImage(frame, 1, (416, 416), (0, 0, 0), True, crop=False) #Blob Detection to identify regions
            
            yolo_net.setInput(blob)
            outputs = yolo_net.forward(out_layers)

            class_no = []
            prediction_probs = []
            boundary_boxes = []
            for out in outputs:
                for detection in out:
                    score = detection[5:]
                    class_id = np.argmax(score)
                    prob = score[class_id]
                    if prob > 0.6:
                        #print(class_no)
                        x_c = int(detection[0] * width)    #creating x coordinate of center
                        y_c = int(detection[1] * height)   #creating y coordinate of center
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(x_c - w / 2)
                        y = int(y_c - h / 2)

                        boundary_boxes.append([x, y, w, h])   #storing information for 1 bounding box in boxes array
                        prediction_probs.append(float(prob)) 
                        class_no.append(class_id)

            indices = cv2.dnn.NMSBoxes(boundary_boxes, prediction_probs, 0.5, 0.4) #non maximum suppression performed given prediction probabilities and corresponding boxes
            
            for i in range(len(boundary_boxes)):
                if i in indices:
                    x, y, w, h = boundary_boxes[i]
                    label = str(classes[class_no[i]])  #insert label
                    color = colors[class_no[i]]  #color of boundry box & text
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) #inserting rectangle as boundary box
                    cv2.putText(frame, label, (x, y + 30), font, 1, color, 2) #1 is the font_scale. 


            cv2.imshow(Name, frame) #display current frame with detections
            key = cv2.waitKey(1)
            if key == 27: #ESC to exit the window
                break
        cam.release()
        cv2.destroyWindow(Name)
     
    elif recording==1 and YOLO==1:
        cv2.namedWindow(Name)
        
        yolo_net = cv2.dnn.readNet("yolov3_face_mask.weights", "mask-yolov3 - Copy.cfg")
        classes = ["no_mask", "mask"]
        names_of_layers = yolo_net.getLayerNames()
        out_layers = [names_of_layers[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cam = cv2.VideoCapture(IPAddress)
        width = int(cam.get(3))
        height = int(cam.get(4))
        size = (width, height)
        outputFile = cv2.VideoWriter(
            Name+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
        
        
        
        while True:
            ret, frame = cam.read()
            frame = cv2.resize(frame, None, fx=1, fy=1)
            height, width, channels = frame.shape

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            yolo_net.setInput(blob)
            outputs = yolo_net.forward(out_layers)

            class_no = []
            prediction_probs = []
            boundary_boxes = []
            for out in outputs:
                for detection in out:
                    score = detection[5:]
                    class_id = np.argmax(score)
                    prob = score[class_id]
                    if prob > 0.6:
                        #print(class_id)
                        x_c = int(detection[0] * width)    #creating x coordinate of center
                        y_c = int(detection[1] * height)   #creating y coordinate of center
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(x_c - w / 2)
                        y = int(y_c - h / 2)

                        boundary_boxes.append([x, y, w, h])   #storing information for 1 bounding box in boxes array
                        prediction_probs.append(float(prob))
                        class_no.append(class_id)

            indices = cv2.dnn.NMSBoxes(boundary_boxes, prediction_probs, 0.5, 0.4) #non maximum suppression performed given prediction probabilities and corresponding boxes
            
            for i in range(len(boundary_boxes)):
                if i in indices:
                    x, y, w, h = boundary_boxes[i]
                    label = str(classes[class_no[i]])  #insert label
                    color = colors[class_no[i]]  #color of boundry box & text
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) #inserting rectangle as boundary box
                    cv2.putText(frame, label, (x, y + 30), font, 1, color, 2) #1 is the font_scale. 

            
            
            
            cv2.imshow(Name, frame)
            key = cv2.waitKey(50)   #Save frame in video after 50ms
            if key == 27: #ESC to exit the window
                break
            
            frame = cv2.resize(frame, size)
            outputFile.write(frame)
            
        cam.release()
        outputFile.release()
        cv2.destroyWindow(Name)    
def main():

    print("Press 1 to record stream, 2 for live stream, 3 for live stream with Mask Detection, 4 to record live stream with Mask Detection: ")
    option = int(input())

    if option == 1:
        # Record video
        thread1 = camThread("Stream1Recording", "http://192.168.100.75:4747/video",recording=1,YOLO=0)
        thread2 = camThread("Stream2Recording", "http://10.130.21.174:8080/video",recording=1,YOLO=0)
        thread3 = camThread("Stream3Recording", "http://10.130.146.242:8080/video",recording=1,YOLO=0)

        thread1.start()
        thread2.start()
        thread3.start()
        print()
        print("Number of current threads: ", threading.activeCount())


    elif option == 2:
        # live stream
        thread1 = camThread("Live Stream of Camera 1", "http://192.168.100.75:4747/video",recording=0,YOLO=0)
        thread2 = camThread("Live Stream of Camera 2", "http://192.168.100.149:8080/video",recording=0,YOLO=0)
        thread3 = camThread("Live Stream of Camera 3", "http://10.130.146.242:8080/video",recording=0,YOLO=0)

        thread1.start()
        thread2.start()
        thread3.start()
        print()
        print("Number of current threads: ", threading.activeCount())

    
    elif option == 3:
        # live stream with Mask Detection
        # in place of IPAddress, if we place path of recorded video, it will perform object detection on that video
        thread1 = camThread("Live Stream of Camera 1", "http://192.168.100.23:8080/video",recording=0,YOLO=1)
        thread2 = camThread("Live Stream of Camera 2", "http://192.168.100.149:8080/video",recording=0,YOLO=1)
        thread3 = camThread("Live Stream of Camera 2", "http://192.168.100.75:4747/video",recording=0,YOLO=1)
        thread1.start()
        thread2.start()
        thread3.start()
        print()
        print("Number of current threads: ", threading.activeCount())
        
    elif option == 4:
        # Record live stream with Mask Detection
        # in place of IPAddress, if we place path of recorded video, it will perform object detection on that video
        thread1 = camThread("Live Stream of Camera 1", "http://192.168.100.23:8080/video",recording=1,YOLO=1)
        thread2 = camThread("Live Stream of Camera 2", "http://192.168.100.149:8080/video",recording=1,YOLO=1)
        thread3 = camThread("Live Stream of Camera 2", "http://192.168.100.75:4747/video",recording=1,YOLO=1)
        thread1.start()
        thread2.start()
        thread3.start()
        print()
        print("Number of current threads: ", threading.activeCount())
    else:
        print("Invalid option entered. Exiting...")


main()
