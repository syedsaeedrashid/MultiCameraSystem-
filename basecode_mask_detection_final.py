import cv2
import threading
import numpy as np
import glob
import random
import imutils
import itertools
import math
from matplotlib import pyplot as plt
#from help_functions import set_points_from_mouse

NMS_THRESHOLD=0.4
MIN_CONFIDENCE=0.1

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
BIG_CIRCLE = 60
SMALL_CIRCLE = 3

heatmap_points = list()
violation_heatmap_points = list()
distance_minimum = 110 #in pixels
skip = 10

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
        
    elif recording==1 and YOLO==0:              #if recording mode is set on, then record the stream, without YOLO Detection
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
            key = cv2.waitKey(1)
            if key == 27:  # exit on ESC
                break
        cam.release()
        outputFile.release()
        cv2.destroyWindow(Name)
          
def draw_circle(event,x,y,flags,param):
    global positions,count
    # If event is Left Button Click then store the coordinate in the lists
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img,(x,y),2,(255,0,0),-1)
        positions.append([x,y])
        if(count!=3):
            positions2.append([x,y])
        elif(count==3):
            positions2.insert(2,[x,y])
        count+=1
        
    

def oneWindow(cam1,cam2,cam3,recording,YOLO):
    cv2.namedWindow("Window")
    if recording==0 and YOLO==0:   #if recording mode is set off, then only stream, without YOLO Detection
        while True:
            ret1, img1 = cam1.read()
            ret2, img2 = cam2.read()
            ret3, img3 = cam3.read()
            if ret1==False or ret2==False or ret3==False:
                print("could not read from cameras !")
                break
            img1 = cv2.resize(img1,(720,480))
            img2 = cv2.resize(img2,(720,480))
            img3 = cv2.resize(img3,(720,480))
            dummy = np.copy(img3)
            dummy[:,:,0] = 0
            dummy[:,:,1] = 0
            dummy[:,:,2] = 0
            final1 = cv2.vconcat([img1,img2])
            final2 = cv2.vconcat([img3,dummy])
            final = cv2.hconcat([final1,final2])
            
            cv2.imshow("Window", final)
            key = cv2.waitKey(1)
            if key == 27:  # exit on ESC
                break
        cam1.release()
        cam2.release()
        cam3.release()
        cv2.destroyWindow("Window")

    elif recording==0 and YOLO==1:   #if recording mode is set on, then record, with YOLO Detection

        # read pretrained network and its weights
        
        #yolo_net = cv2.dnn.readNet("mask-yolov4-tiny_best.weights", "mask-yolov4-tiny.cfg")
        yolo_net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
        #classes = ["no_mask", "mask"]
        classes = ["person"]
        names_of_layers = yolo_net.getLayerNames()
        out_layers = [names_of_layers[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3)) #randomly assigning x separate colors for x classes 

        font = cv2.FONT_HERSHEY_SIMPLEX     #Simplex font is usually the most thin, hence used this

        while True:
            ret1, img1 = cam1.read()
            ret2, img2 = cam2.read()
            ret3, img3 = cam3.read()
            if ret1==False or ret2==False or ret3==False:
                print("could not read from cameras !")
                break
            
            img1 = cv2.resize(img1,(720,480))
            img2 = cv2.resize(img2,(720,480))
            img3 = cv2.resize(img3,(720,480))
            dummy = np.copy(img3)
            dummy[:,:,0] = 0
            dummy[:,:,1] = 0
            dummy[:,:,2] = 0
            final1 = cv2.vconcat([img1,img2])
            final2 = cv2.vconcat([img3,dummy])
            final = cv2.hconcat([final1,final2])
            
            frame = cv2.resize(final, None, fx=1, fy=1) 
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


            cv2.imshow("Window", frame) #display current frame with detections
            key = cv2.waitKey(1)
            if key == 27: #ESC to exit the window
                break

        cam1.release()
        cam2.release()
        cam3.release()
        cv2.destroyWindow("Window")

def boxes_detection(image, model, layer_name,labels):
    (H, W) = image.shape[:2]
    results = []


    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)


    centroids = [] ; boxes = []
    confidences = []
    class_no = []; colors = []; colors.append((0,255,0)) #Masked
    colors.append((0,0,255)) #UnMasked
    colors.append((0,0,255))
    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if  confidence > MIN_CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
                class_no.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
    # ensure at least one detection exists
    #if len(idzs) > 0:
    Boxes_to_keep = []
    for i in range(len(boxes)):
        # loop over the indexes we are keeping
        if i in idzs:
            # extract the bounding box coordinates
            #Boxes_to_keep.append(boxes[i])
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            Boxes_to_keep.append((x, y, x + w, y + h))
            color=colors[class_no[i]]
            label = str(labels[class_no[i]])  #insert label
            res = (confidences[i], (x, y, x + w, y + h), centroids[i],color,label)
            #res = (confidences[i], (x, y, x + w, y + h), centroids[i])
           

            results.append(res)
    # return the list of results
    return results,Boxes_to_keep

def get_points_from_box(box):
    
    # Center of the box x = (x1+x2)/2 et y = (y1+y2)/2
    #print(box)
    center_x = int(((box[1]+box[3])/2))
    center_y = int(((box[0]+box[2])/2))
    #center_x = int(((box[1]+box[3])/2))
    #center_y = int(((box[0]+box[2])/2))
    # Coordiniate on the point at the bottom center of the box
    center_y_ground = center_y + ((box[2] - box[0])/2)
    return (center_x,center_y),(center_x,int(center_y_ground))


def get_centroids_and_groundpoints(array_boxes_detected):
	
	array_centroids,array_groundpoints = [],[] # Initialize empty centroid and ground point lists 
	for index,box in enumerate(array_boxes_detected):
		# Draw the bounding box 
		# c
		# Get the both important points
		centroid,ground_point = get_points_from_box(box)
		array_centroids.append(centroid)
		array_groundpoints.append(ground_point)
	return array_centroids,array_groundpoints


def process(IPAddress,Name):
    
    
    labelsPath = "mask-tiny.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    weights_path = "mask-tiny_last.weights"
    config_path = "mask-tiny.cfg"
    #weights_path = "yolov3-tiny.weights"
    #config_path = "yolov3-tiny.cfg"
    
    
    model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
    #model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)



    layer_name = model.getLayerNames()
    layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()] ; colors = []
        
    cap1 = cv2.VideoCapture(IPAddress)
    writer = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        (grabbed1, image1) = cap1.read()
        if not grabbed1:
            break
        image1 = imutils.resize(image1, width=700)
        results,_ = boxes_detection(image1, model, layer_name,LABELS)

        for res in results:
            cv2.rectangle(image1, (res[1][0],res[1][1]), (res[1][2],res[1][3]), res[3], 2); cv2.putText(image1, res[4], (res[1][0], res[1][1] + 30), font, 0.5, res[3], 2) #0.5 is the font_scale.
        cv2.imshow(Name,image1)


        key = cv2.waitKey(1)
        if key == 27:
            break

    cap1.release()
    cv2.destroyAllWindows()



def compute_homographies():
    topview = cv2.imread("Topview.png")
    camera1_img = cv2.imread("1_snapshot.png")
    camera2_img = cv2.imread("2_snapshot.png")
    camera3_img = cv2.imread("3_snapshot.png")

    '''camera1_img = cv2.imread("Stream1Recording_Moment.jpg")
    camera2_img = cv2.imread("Stream2Recording_Moment.jpg")
    camera3_img = cv2.imread("Stream3Recording_Moment.jpg")'''
    #camera1 homography
    pts =        np.array( [[651, 685], [796, 664],[667,814] ,[1047, 755],[1456,819],[1075,916],[799,653],[1258,653],[1082,951]] ).astype(np.float32)
    target_pts = np.array( [[654, 724], [653, 681],[954,659] ,[874, 583],[954,508],[1026,583],[640,671],[640,495],[1046,581]] ).astype(np.float32)
    '''
    pts =        np.array( [[323,425],[469,452],[479,429],[277,448],[259,458]] ).astype(np.float32)
    target_pts = np.array( [[1222,585],[1064,585],[1147,663],[1147,508],[1120,488]] ).astype(np.float32)
    '''
    M1, status1 = cv2.findHomography(pts, target_pts)

    #camera2 homography
    
    pts =        np.array( [[1367,874],[1740,913],[1015,913],[1622,968],[1622,915],[1881,949],[1365,932],[1293,936],[1089,1003],[547,982],[918,955],[644,1045]] ).astype(np.float32)
    target_pts = np.array( [[1435,586],[1340,725],[1340,439],[1167,677],[1309,677],[1246,756],[1246,607],[1219,582],[1048,567],[1048,419],[1124,487],[962,487]] ).astype(np.float32)
    
    '''
    pts =        np.array( [[295,365],[292,458],[410,409],[179,398]] ).astype(np.float32)
    target_pts = np.array( [[1064,587],[1217,587],[1139,513],[1139,656]] ).astype(np.float32)
    '''
    M2, status2 = cv2.findHomography(pts, target_pts)

    #camera3 homography
    
    pts =        np.array( [[531,1000],[174,1034],[901,1015],[817,1025],[956,1072],[1074,1041],[677,1055]] ).astype(np.float32)
    target_pts = np.array( [[1438,590],[1336,447],[1336,725],[1309,677],[1167,677],[1246,756],[1246,607]] ).astype(np.float32)
    
    '''
    pts =        np.array( [[367,435],[162,446],[185,491],[446,474]] ).astype(np.float32)
    target_pts = np.array( [[1126,680],[1047,606],[971,680],[1047,757]] ).astype(np.float32)
    '''
    M3, status3 = cv2.findHomography(pts, target_pts)


    return M1,M2,M3

def compute_face_homographies():
    topview = cv2.imread("Topview_g.png")
    camera1_img = cv2.imread("1g.png")
    #camera2_img = cv2.imread("2_recorded_Moment.jpg")
    #camera3_img = cv2.imread("3_recorded_Moment.jpg")

    
    #camera1 homography
    '''pts =        np.array( [[1001,551],[1643,463],[1329,491],[881,415],[1873,471]] ).astype(np.float32)
    target_pts = np.array( [[1269,576],[973,490],[771,485],[590,636],[922,436]] ).astype(np.float32)'''

    pts =        np.array( [[1381,452],[903,469],[1073,458],[108,472],[206,465],[710,470],[1418,520],[601,479],[874,463],[322,472],[869,499],[690,492],[524,502],[442,480],[628,479]] ).astype(np.float32)
    target_pts = np.array( [[377,344],[284,116],[342,30],[81,289],[106,273],[236,370],[393,269],[252,449],[270,53],[180,380],[272,344],[229,242],[195,307],[158,254],[235,406]] ).astype(np.float32)
    
    M1, status1 = cv2.findHomography(pts, target_pts)
    '''
    #camera2 homography
    
    pts =        np.array( [[535,723],[1167,723],[287,751],[845,745]] ).astype(np.float32)
    target_pts = np.array( [[1022,412],[1249,541],[991,366],[1246,388]] ).astype(np.float32)
    M2, status2 = cv2.findHomography(pts, target_pts)

    #camera3 homography
    
    pts =        np.array( [[112,860],[1238,841],[1638,831],[1298,888],[427,858]] ).astype(np.float32)
    target_pts = np.array( [[892,467],[794,575],[818,604],[819,584],[1200,548]] ).astype(np.float32)
    
    M3, status3 = cv2.findHomography(pts, target_pts)'''


    #return M1,M2,M3
    return M1

def get_perspective_transform(M,frame,topview):
    
    transformed_camera = cv2.warpPerspective(frame, M, (topview.shape[1],topview.shape[0]))
    return transformed_camera

def compute_point_perspective_transformation(matrix,list_downoids,topview):
	
    # Compute the new coordinates of our points
    list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)

    # Loop over the points and add them to the list that will be returned
    transformed_points_list = list()
    if transformed_points is None:
        return transformed_points_list
    else:
        for i in range(0,transformed_points.shape[0]):
            transformed_points_list.append([transformed_points[i][0][0],transformed_points[i][0][1]])
    return transformed_points_list


def get_human_box_detection(boxes,scores,classes,height,width):
    
    array_boxes = list() # Create an empty list
    for i in range(boxes.shape[1]):
        # If the class of the detected object is 1 and the confidence of the prediction is > 0.6
        if int(classes[i]) == 1 and scores[i] > 0.75:
            # Multiply the X coordonnate by the height of the image and the Y coordonate by the width
            # To transform the box value into pixel coordonate values.
            box = [boxes[0,i,0],boxes[0,i,1],boxes[0,i,2],boxes[0,i,3]] * np.array([height, width, height, width])
            # Add the results converted to int
            array_boxes.append((int(box[0]),int(box[1]),int(box[2]),int(box[3])))
    return array_boxes

def set_points_from_mouse(stream,frame_name,frame_size):
# Define the callback function that we are going to use to get our coordinates
        def CallBackFunc(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print("Left button of the mouse is clicked - position (", x, ", ",y, ")")
                list_points.append([x,y])
            elif event == cv2.EVENT_RBUTTONDOWN:
                print("Right button of the mouse is clicked - position (", x, ", ", y, ")")
                list_points.append([x,y])

        size_frame = frame_size
        vs = cv2.VideoCapture(stream)
        # Loop until the end of the video stream
        while True:    
            # Load the frame and test if it has reache the end of the video
            (frame_exists, frame) = vs.read()
            frame = imutils.resize(frame, width=int(size_frame))
            cv2.imwrite(frame_name,frame)
            break

        # Create a black image and a window
        windowName = 'MouseCallback'
        cv2.namedWindow(windowName)


        # Load the image 
        img_path = frame_name
        img = cv2.imread(img_path)

        
        # Get the size of the image for the calibration
        width,height,_ = img.shape

        # Create an empty list of points for the coordinates
        
        list_points = list()
        # bind the callback function to window
        cv2.setMouseCallback(windowName, CallBackFunc)
        points = []
        if __name__ == "__main__":
        # Check if the 4 points have been saved
            while (True):
                cv2.imshow(windowName, img)
                if len(list_points) == 4:
                    p2 = list_points[3]
                    p1 = list_points[2]
                    p4 = list_points[0]
                    p3 = list_points[1]
                    points.append(p1);  points.append(p2);  points.append(p3);  points.append(p4)
                    break

                if cv2.waitKey(20) == 27:
                    break
        
        cv2.destroyAllWindows()
        return points
def change_color_on_topview(pair,top_view):
    """
    Draw red circles for the designated pair of points 
    """
    cv2.circle(top_view, (int(pair[0][0]),int(pair[0][1])), BIG_CIRCLE, COLOR_RED, 2)
    cv2.circle(top_view, (int(pair[0][0]),int(pair[0][1])), SMALL_CIRCLE, COLOR_RED, -1)
    cv2.circle(top_view, (int(pair[1][0]),int(pair[1][1])), BIG_CIRCLE, COLOR_RED, 2)
    cv2.circle(top_view, (int(pair[1][0]),int(pair[1][1])), SMALL_CIRCLE, COLOR_RED, -1)

def set_points_from_mouse_topview(stream,frame_name):
# Define the callback function that we are going to use to get our coordinates
        def CallBackFunc(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print("Left button of the mouse is clicked - position (", x, ", ",y, ")")
                list_points.append([x,y])
            elif event == cv2.EVENT_RBUTTONDOWN:
                print("Right button of the mouse is clicked - position (", x, ", ", y, ")")
                list_points.append([x,y])

       
        windowName = 'MouseCallback'
        cv2.namedWindow(windowName)
        cv2.namedWindow(frame_name)


        # Load the image 
        img_path = stream
        img = cv2.imread(img_path)
        frame = cv2.imread(frame_name)
        frame = cv2.resize(frame,(720,480))
        
        # Get the size of the image for the calibration
        width,height,_ = img.shape

        # Create an empty list of points for the coordinates
        
        list_points = list()
        # bind the callback function to window
        cv2.setMouseCallback(windowName, CallBackFunc)
        points = []
        if __name__ == "__main__":
        # Check if the 4 points have been saved
            while (True):
                cv2.imshow(windowName, img)
                cv2.imshow(frame_name,frame)
                if len(list_points) == 4:
                    p2 = list_points[3]
                    p1 = list_points[2]
                    p4 = list_points[0]
                    p3 = list_points[1]
                    points.append(p1);  points.append(p2);  points.append(p3);  points.append(p4)
                    break

                if cv2.waitKey(20) == 27:
                    break
        
        cv2.destroyAllWindows()
        return points


def main():

    IP1 = "http://10.130.10.242:4747/video"; IP2 = "http://10.130.22.33:8080/video"; IP3 = 1

    print("Press 1 to record stream, 2 for live stream, 3 for live stream in one window , 4 to live stream with Mask Detection, 5 to detect Masks on pre recorded videos, 6 to display top view, 7 to project detections on top view with SOP Violations, 8 to view animated heatmap, 9 to view violations heatmap:")
    option = int(input())

    if option == 1:
        # Record video
        thread1 = camThread("StreamRecording1", IP1,recording=1,YOLO=0)
        thread2 = camThread("StreamRecording2", IP2,recording=1,YOLO=0)
        thread3 = camThread("StreamRecording3", IP3,recording=1,YOLO=0)

        thread1.start()
        thread2.start()
        thread3.start()
        print()
        print("Number of current threads: ", threading.activeCount())


    elif option == 2:
        # live stream
        thread1 = camThread("Live Stream of Camera 1", IP1,recording=0,YOLO=0)
        thread2 = camThread("Live Stream of Camera 2", IP2,recording=0,YOLO=0)
        thread3 = camThread("Live Stream of Camera 3", IP3,recording=0,YOLO=0)

        thread1.start()
        thread2.start()
        thread3.start()
        print()
        print("Number of current threads: ", threading.activeCount())

    elif option ==3:
        #live stream of three videos in one window
        cam1 = cv2.VideoCapture(IP1)
        cam2 = cv2.VideoCapture(IP2)
        cam3 = cv2.VideoCapture(IP3)
        oneWindow(cam1,cam2,cam3,0,0)

    elif option == 4:

        t1 = threading.Thread(target=process,args=(IP1,"Stream1"))
        t2 = threading.Thread(target=process,args=(IP2,"Stream2"))
        t3 = threading.Thread(target=process,args=(IP3,"Stream3"))

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()
        
        
        
        
    elif option == 5:
       
        # in place of IPAddress, if we place path of recorded video, it will perform object detection on that video
        video_path1 = "1_recorded.mp4"; video_path2="2_recorded.mp4" ; video_path3="3_recorded.mp4"
        #video_path1 = "1g.mp4"; video_path2="2g.mp4" ; video_path3="3g.mp4"
        #video_path1 = "Stream1Recording.avi"; video_path2="Stream2Recording.avi" ; video_path3="Stream3Recording.avi"
        #video_path1 = "trim_1_vid_1.avi"; video_path2 = "trim_2_vid_1.avi" ; video_path3 = "trim_3_vid_1.avi"

        t1 = threading.Thread(target=process,args=(video_path1,"Stream1"))
        t2 = threading.Thread(target=process,args=(video_path2,"Stream2"))
        t3 = threading.Thread(target=process,args=(video_path3,"Stream3"))

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()
    

    elif option==6 :

        video_path1 = "1_recorded.mp4"; video_path2="2_recorded.mp4" ; video_path3="3_recorded.mp4"; Name = "TopView"
        #video_path1 = "Stream1Recording.avi"; video_path2="Stream2Recording.avi" ; video_path3="Stream3Recording.avi"; Name = "TopView Detection"
        M1,M2,M3 = compute_homographies()
        topview = cv2.imread("Topview.png")

        


        #top_view_generation from frames
        cv2.namedWindow(Name)
        cv2.namedWindow("1_view")
        cv2.namedWindow("2_view")
        cv2.namedWindow("3_view")

        cam1 = cv2.VideoCapture(video_path1)
        if cam1.isOpened():
            ret1, frame1 = cam1.read()
        else:
            ret1 = False
        cam2 = cv2.VideoCapture(video_path2)

        if cam2.isOpened():
            ret2, frame2 = cam2.read()
        else:
            ret2 = False

        cam3 = cv2.VideoCapture(video_path3)
        if cam3.isOpened():
            ret3, frame3 = cam3.read()
        else:
            ret3 = False


        while ret1 and ret2 and ret3:
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()
            ret3, frame3 = cam3.read()

            transformed_camera1 = get_perspective_transform(M1,frame1,topview)
            transformed_camera2 = get_perspective_transform(M2,frame2,topview)
            transformed_camera3 = get_perspective_transform(M3,frame3,topview)

            #stitching
            out_temp = cv2.addWeighted(transformed_camera1, 0.5,transformed_camera2 , 0.5, 0)
            out = cv2.addWeighted(out_temp, 0.5, transformed_camera3 , 0.5, 0)
            #stitched_on_topview = cv2.addWeighted(topview, 0.9, out , 0.2, 0)
            
            out = cv2.resize(out,(720,480))
            transformed_camera1 = cv2.resize(transformed_camera1,(480,320))
            transformed_camera2 = cv2.resize(transformed_camera2,(480,320))
            transformed_camera3 = cv2.resize(transformed_camera3,(480,320))
            cv2.imshow(Name, out)
            cv2.imshow("1_view", transformed_camera1)
            cv2.imshow("2_view", transformed_camera2)
            cv2.imshow("3_view", transformed_camera3)
            key = cv2.waitKey(1)
            if key == 27:  # exit on ESC
                break
        cam1.release()
        cam2.release()
        cam3.release()
        cv2.destroyWindow(Name)
        cv2.destroyallwindows()
    
    elif option==7:

        
        font = cv2.FONT_HERSHEY_SIMPLEX
        #video_path1 = "1_recorded.mp4"; video_path2="2_recorded.mp4" ; video_path3="3_recorded.mp4"; Name = "TopView Detection"
        video_path1 = "vlc-record-2021-12-24-08h48m55s-1g.mp4-.mp4"; video_path2="vlc-record-2021-12-24-08h55m23s-2g.mp4-.mp4" ; video_path3="vlc-record-2021-12-24-09h00m15s-3g.mp4-.mp4"; Name = "TopView Detection"
        #video_path1 = "Stream1Recording.avi"; video_path2="Stream2Recording.avi" ; video_path3="Stream3Recording.avi"; Name = "TopView Detection"
        labelsPath = "mask-tiny.names"
        LABELS = open(labelsPath).read().strip().split("\n")
        topview = cv2.imread("Topview_g.png")
        M1,M2,M3 = compute_homographies()

        #Mf1,Mf2,Mf3 = compute_face_homographies()
        Mf1 = compute_face_homographies()
        weights_path = "mask-tiny_last.weights"
        config_path = "mask-tiny.cfg"
        #weights_path = "yolov3-tiny.weights"
        #config_path = "yolov3-tiny.cfg"
        model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        

        layer_name = model.getLayerNames()
        layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()] 

        #top_view_generation from frames
        cv2.namedWindow(Name)
        cv2.namedWindow("1")
        cv2.namedWindow("2")
        cv2.namedWindow("3")

      

        cam1 = cv2.VideoCapture(video_path1)
        if cam1.isOpened():
            ret1, frame1 = cam1.read()
        else:
            ret1 = False
        cam2 = cv2.VideoCapture(video_path2)

        if cam2.isOpened():
            ret2, frame2 = cam2.read()
        else:
            ret2 = False

        cam3 = cv2.VideoCapture(video_path3)
        if cam3.isOpened():
            ret3, frame3 = cam3.read()
        else:
            ret3 = False

        while ret1 and ret2 and ret3:
            topview = cv2.imread("Topview_g.png")
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()
            ret3, frame3 = cam3.read()
            
            frame1 = imutils.resize(frame1, width=700)
            results1,boxes1 = boxes_detection(frame1, model, layer_name,LABELS)
            frame2 = imutils.resize(frame2, width=700)
            results2,boxes2 = boxes_detection(frame2, model, layer_name,LABELS)
            frame3 = imutils.resize(frame3, width=700)
            results3,boxes3 = boxes_detection(frame3, model, layer_name,LABELS)

            for res in results1:
                cv2.rectangle(frame1, (res[1][0],res[1][1]), (res[1][2],res[1][3]), res[3], 2); cv2.putText(frame1, res[4], (res[1][0], res[1][1] + 30), font, 0.5, res[3], 2) #0.5 is the font_scale.
            for res in results2:
                cv2.rectangle(frame2, (res[1][0],res[1][1]), (res[1][2],res[1][3]), res[3], 2); cv2.putText(frame2, res[4], (res[1][0], res[1][1] + 30), font, 0.5, res[3], 2) #0.5 is the font_scale.
            for res in results3:
                cv2.rectangle(frame3, (res[1][0],res[1][1]), (res[1][2],res[1][3]), res[3], 2); cv2.putText(frame3, res[4], (res[1][0], res[1][1] + 30), font, 0.5, res[3], 2) #0.5 is the font_scale.
        
            
            array_centroids1,array_groundpoints1 = get_centroids_and_groundpoints(boxes1)
            array_centroids2,array_groundpoints2 = get_centroids_and_groundpoints(boxes2)
            array_centroids3,array_groundpoints3 = get_centroids_and_groundpoints(boxes3)

            transformed_points1 = compute_point_perspective_transformation(Mf1,array_groundpoints1,topview)
            #transformed_points2 = compute_point_perspective_transformation(Mf2,array_groundpoints2,topview)
            #transformed_points3 = compute_point_perspective_transformation(Mf3,array_groundpoints3,topview)

            #transformed_camera1 = get_perspective_transform(M1,frame1,topview)

            for point in transformed_points1:
                x,y = point
                cv2.circle(topview, (int(x),int(y)), BIG_CIRCLE, COLOR_GREEN, 2)
                cv2.circle(topview, (int(x),int(y)), SMALL_CIRCLE, COLOR_GREEN, -1)
            '''
            for point in transformed_points2:
                x,y = point
                cv2.circle(topview, (int(x),int(y)), BIG_CIRCLE, COLOR_GREEN, 2)
                cv2.circle(topview, (int(x),int(y)), SMALL_CIRCLE, COLOR_GREEN, -1)
            
            for point in transformed_points3:
                x,y = point
                cv2.circle(topview, (int(x),int(y)), BIG_CIRCLE, COLOR_GREEN, 2)
                cv2.circle(topview, (int(x),int(y)), SMALL_CIRCLE, COLOR_GREEN, -1)
            '''
            width = topview.shape[1]
            height= topview.shape[0]
            # Check if 2 or more people have been detected
            if len(transformed_points1) >= 2:
            
                # Iterate over every possible 2 by 2 between the points combinations 
                list_indexes = list(itertools.combinations(range(len(transformed_points1)), 2))
                for i,pair in enumerate(itertools.combinations(transformed_points1, r=2)):
                    # Check if the distance between each combination of points is less than the minimum distance chosen
                    if math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 ) < int(distance_minimum):
                        # Change the colors of the points that are too close from each other to red
                        if not (pair[0][0] > width or pair[0][0] < 0 or pair[0][1] > height+200  or pair[0][1] < 0 or pair[1][0] > width or pair[1][0] < 0 or pair[1][1] > height+200  or pair[1][1] < 0):
                            change_color_on_topview(pair,topview)
                            index_pt1 = list_indexes[i][0]
                            index_pt2 = list_indexes[i][1]
                            cv2.rectangle(frame1, (results1[index_pt1][1][0],results1[index_pt1][1][1]), (results1[index_pt1][1][2],results1[index_pt1][1][3]), COLOR_BLUE, 2)
                            cv2.rectangle(frame1, (results1[index_pt2][1][0],results1[index_pt2][1][1]), (results1[index_pt2][1][2],results1[index_pt2][1][3]), COLOR_BLUE, 2)
            
                            


            #topview = cv2.resize(topview,(720,480))
            frame1 = cv2.resize(frame1,(720,480))
            frame2 = cv2.resize(frame2,(720,480))
            frame3 = cv2.resize(frame3,(720,480))
            cv2.imshow(Name, topview)
            cv2.imshow("1", frame1)
            cv2.imshow("2", frame2)
            cv2.imshow("3", frame3)
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
        cam1.release()
        cam2.release()
        cam3.release()
        cv2.destroyWindow(Name)
        cv2.destroyWindow("1")
        cv2.destroyWindow("2")
        cv2.destroyWindow("3")

    elif option == 8:

        font = cv2.FONT_HERSHEY_SIMPLEX
        #video_path1 = "1_recorded.mp4"; video_path2="2_recorded.mp4" ; video_path3="3_recorded.mp4"; Name = "TopView Detection"
        video_path1 = "1g.mp4"; video_path2="2g.mp4" ; video_path3="3g.mp4"; Name = "HeatMap"
        #video_path1 = "Stream1Recording.avi"; video_path2="Stream2Recording.avi" ; video_path3="Stream3Recording.avi"; Name = "TopView Detection"
        labelsPath = "mask-tiny.names"
        LABELS = open(labelsPath).read().strip().split("\n")
        topview = cv2.imread("Topview_g.png")
        M1,M2,M3 = compute_homographies()

        #Mf1,Mf2,Mf3 = compute_face_homographies()
        Mf1 = compute_face_homographies()
        weights_path = "mask-tiny_last.weights"
        config_path = "mask-tiny.cfg"
        #weights_path = "yolov3-tiny.weights"
        #config_path = "yolov3-tiny.cfg"
        model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        

        layer_name = model.getLayerNames()
        layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()] 

        #top_view_generation from frames
        cv2.namedWindow(Name)
        cv2.namedWindow("1")
        cv2.namedWindow("2")
        cv2.namedWindow("3")

      

        cam1 = cv2.VideoCapture(video_path1)
        if cam1.isOpened():
            ret1, frame1 = cam1.read()
        else:
            ret1 = False
        cam2 = cv2.VideoCapture(video_path2)

        if cam2.isOpened():
            ret2, frame2 = cam2.read()
        else:
            ret2 = False

        cam3 = cv2.VideoCapture(video_path3)
        if cam3.isOpened():
            ret3, frame3 = cam3.read()
        else:
            ret3 = False

        while ret1 and ret2 and ret3:
            topview = cv2.imread("Topview_g.png")
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()
            ret3, frame3 = cam3.read()
            
            frame1 = imutils.resize(frame1, width=700)
            results1,boxes1 = boxes_detection(frame1, model, layer_name,LABELS)
            frame2 = imutils.resize(frame2, width=700)
            results2,boxes2 = boxes_detection(frame2, model, layer_name,LABELS)
            frame3 = imutils.resize(frame3, width=700)
            results3,boxes3 = boxes_detection(frame3, model, layer_name,LABELS)

            for res in results1:
                cv2.rectangle(frame1, (res[1][0],res[1][1]), (res[1][2],res[1][3]), res[3], 2); cv2.putText(frame1, res[4], (res[1][0], res[1][1] + 30), font, 0.5, res[3], 2) #0.5 is the font_scale.
            for res in results2:
                cv2.rectangle(frame2, (res[1][0],res[1][1]), (res[1][2],res[1][3]), res[3], 2); cv2.putText(frame2, res[4], (res[1][0], res[1][1] + 30), font, 0.5, res[3], 2) #0.5 is the font_scale.
            for res in results3:
                cv2.rectangle(frame3, (res[1][0],res[1][1]), (res[1][2],res[1][3]), res[3], 2); cv2.putText(frame3, res[4], (res[1][0], res[1][1] + 30), font, 0.5, res[3], 2) #0.5 is the font_scale.
        
            
            array_centroids1,array_groundpoints1 = get_centroids_and_groundpoints(boxes1)
            array_centroids2,array_groundpoints2 = get_centroids_and_groundpoints(boxes2)
            array_centroids3,array_groundpoints3 = get_centroids_and_groundpoints(boxes3)

            transformed_points1 = compute_point_perspective_transformation(Mf1,array_groundpoints1,topview)
            #transformed_points2 = compute_point_perspective_transformation(Mf2,array_groundpoints2,topview)
            #transformed_points3 = compute_point_perspective_transformation(Mf3,array_groundpoints3,topview)

            #transformed_camera1 = get_perspective_transform(M1,frame1,topview)
            
            for point in transformed_points1:
                x,y = point
                if (x >=0 and y>=0):
                    heatmap_points.append(point)
    
            '''
            for point in transformed_points2:
                x,y = point
                cv2.circle(topview, (int(x),int(y)), BIG_CIRCLE, COLOR_GREEN, 2)
                cv2.circle(topview, (int(x),int(y)), SMALL_CIRCLE, COLOR_GREEN, -1)
            
            for point in transformed_points3:
                x,y = point
                cv2.circle(topview, (int(x),int(y)), BIG_CIRCLE, COLOR_GREEN, 2)
                cv2.circle(topview, (int(x),int(y)), SMALL_CIRCLE, COLOR_GREEN, -1)
            '''
            width = topview.shape[1]
            height= topview.shape[0]
            # Check if 2 or more people have been detected
            if len(transformed_points1) >= 2:
                
                # Iterate over every possible 2 by 2 between the points combinations 
                list_indexes = list(itertools.combinations(range(len(transformed_points1)), 2))
                for i,pair in enumerate(itertools.combinations(transformed_points1, r=2)):
                    # Check if the distance between each combination of points is less than the minimum distance chosen
                    if math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 ) < int(distance_minimum):
                        # Change the colors of the points that are too close from each other to red
                        if not (pair[0][0] > width or pair[0][0] < 0 or pair[0][1] > height+200  or pair[0][1] < 0 or pair[1][0] > width or pair[1][0] < 0 or pair[1][1] > height+200  or pair[1][1] < 0):
                            #change_color_on_topview(pair,topview)
                            index_pt1 = list_indexes[i][0]
                            index_pt2 = list_indexes[i][1]
                            cv2.rectangle(frame1, (results1[index_pt1][1][0],results1[index_pt1][1][1]), (results1[index_pt1][1][2],results1[index_pt1][1][3]), COLOR_BLUE, 2)
                            cv2.rectangle(frame1, (results1[index_pt2][1][0],results1[index_pt2][1][1]), (results1[index_pt2][1][2],results1[index_pt2][1][3]), COLOR_BLUE, 2)
                            heatmap_points.append(pair[0]); heatmap_points.append(pair[1])
                            
            #setting up gaussian kernel
            k = 21
            gauss = cv2.getGaussianKernel(k,np.sqrt(64))
            gauss = gauss*gauss.T
            gauss = (gauss/gauss[int(k/2),int(k/2)])                
            #blank image for heatmap points initialization
            img2 = np.zeros((topview.shape[0],topview.shape[1],3)).astype(np.float32)
            j = cv2.applyColorMap(((gauss)*255).astype(np.uint8),cv2.COLORMAP_JET).astype(np.float32)/255
            heatmap_points_k = heatmap_points[-skip:]
            
            for p in heatmap_points_k:
                
                if(p[0]-int(k/2)>=0 and p[1]-int(k/2)>=0 and (p[0]+int(k/2))<topview.shape[0] and (p[1]+int(k/2))<topview.shape[1]):
                    
                    p[0] = p[0].astype(int); p[1] = p[1].astype(int)
                    b = img2[p[0]-int(k/2):(p[0]+int(k/2)+1)%topview.shape[0],p[1]-int(k/2):(p[1]+int(k/2)+1)%topview.shape[1],:]
                    c = j+b
                    img2[p[0]-int(k/2):(p[0]+int(k/2)+1)%topview.shape[0],p[1]-int(k/2):(p[1]+int(k/2)+1)%topview.shape[1],:]=c
            
            
            #inverse mask
            g = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            mask1 = np.where(g>0,1,0).astype(np.float32)
            mask_3 = np.ones((topview.shape[0],topview.shape[1],3))*(1-mask1)[:,:,None]
            mask_3 = mask_3/255
            mask = np.where(g>0,255,0).astype(np.float32)
            #multiply by mask
            mask_4=img2*(mask)[:,:,None]
            #generate new_topview
            new_topview = mask_3*topview
            #create heatmap
            heatmap = new_topview + mask_4/255

            
            
            #topview = cv2.resize(topview,(720,480))
            frame1 = cv2.resize(frame1,(720,480))
            frame2 = cv2.resize(frame2,(720,480))
            frame3 = cv2.resize(frame3,(720,480))
            #cv2.imshow(Name, topview)
            cv2.imshow(Name,heatmap)
            cv2.imshow("1", frame1)
            cv2.imshow("2", frame2)
            cv2.imshow("3", frame3)
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
        cam1.release()
        cam2.release()
        cam3.release()
        cv2.destroyWindow(Name)
        cv2.destroyWindow("1")
        cv2.destroyWindow("2")
        cv2.destroyWindow("3") 

    elif option == 9:

        font = cv2.FONT_HERSHEY_SIMPLEX
        #video_path1 = "1_recorded.mp4"; video_path2="2_recorded.mp4" ; video_path3="3_recorded.mp4"; Name = "TopView Detection"
        video_path1 = "1g.mp4"; video_path2="2g.mp4" ; video_path3="3g.mp4"; Name = "HeatMap"
        #video_path1 = "Stream1Recording.avi"; video_path2="Stream2Recording.avi" ; video_path3="Stream3Recording.avi"; Name = "TopView Detection"
        labelsPath = "mask-tiny.names"
        LABELS = open(labelsPath).read().strip().split("\n")
        topview = cv2.imread("Topview_g.png")
        M1,M2,M3 = compute_homographies()

        #Mf1,Mf2,Mf3 = compute_face_homographies()
        Mf1 = compute_face_homographies()
        weights_path = "mask-tiny_last.weights"
        config_path = "mask-tiny.cfg"
        #weights_path = "yolov3-tiny.weights"
        #config_path = "yolov3-tiny.cfg"
        model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        

        layer_name = model.getLayerNames()
        layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()] 

        #top_view_generation from frames
        cv2.namedWindow(Name)
        cv2.namedWindow("1")
        cv2.namedWindow("2")
        cv2.namedWindow("3")

      

        cam1 = cv2.VideoCapture(video_path1)
        if cam1.isOpened():
            ret1, frame1 = cam1.read()
        else:
            ret1 = False
        cam2 = cv2.VideoCapture(video_path2)

        if cam2.isOpened():
            ret2, frame2 = cam2.read()
        else:
            ret2 = False

        cam3 = cv2.VideoCapture(video_path3)
        if cam3.isOpened():
            ret3, frame3 = cam3.read()
        else:
            ret3 = False

        while ret1 and ret2 and ret3:
            topview = cv2.imread("Topview_g.png")
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()
            ret3, frame3 = cam3.read()
            
            frame1 = imutils.resize(frame1, width=700)
            results1,boxes1 = boxes_detection(frame1, model, layer_name,LABELS)
            frame2 = imutils.resize(frame2, width=700)
            results2,boxes2 = boxes_detection(frame2, model, layer_name,LABELS)
            frame3 = imutils.resize(frame3, width=700)
            results3,boxes3 = boxes_detection(frame3, model, layer_name,LABELS)

            for res in results1:
                cv2.rectangle(frame1, (res[1][0],res[1][1]), (res[1][2],res[1][3]), res[3], 2); cv2.putText(frame1, res[4], (res[1][0], res[1][1] + 30), font, 0.5, res[3], 2) #0.5 is the font_scale.
            for res in results2:
                cv2.rectangle(frame2, (res[1][0],res[1][1]), (res[1][2],res[1][3]), res[3], 2); cv2.putText(frame2, res[4], (res[1][0], res[1][1] + 30), font, 0.5, res[3], 2) #0.5 is the font_scale.
            for res in results3:
                cv2.rectangle(frame3, (res[1][0],res[1][1]), (res[1][2],res[1][3]), res[3], 2); cv2.putText(frame3, res[4], (res[1][0], res[1][1] + 30), font, 0.5, res[3], 2) #0.5 is the font_scale.
        
            
            array_centroids1,array_groundpoints1 = get_centroids_and_groundpoints(boxes1)
            array_centroids2,array_groundpoints2 = get_centroids_and_groundpoints(boxes2)
            array_centroids3,array_groundpoints3 = get_centroids_and_groundpoints(boxes3)

            transformed_points1 = compute_point_perspective_transformation(Mf1,array_groundpoints1,topview)
            #transformed_points2 = compute_point_perspective_transformation(Mf2,array_groundpoints2,topview)
            #transformed_points3 = compute_point_perspective_transformation(Mf3,array_groundpoints3,topview)

            #transformed_camera1 = get_perspective_transform(M1,frame1,topview)
            
            for point in transformed_points1:
                x,y = point
                if (x >=0 and y>=0):
                    heatmap_points.append(point)
    
            '''
            for point in transformed_points2:
                x,y = point
                cv2.circle(topview, (int(x),int(y)), BIG_CIRCLE, COLOR_GREEN, 2)
                cv2.circle(topview, (int(x),int(y)), SMALL_CIRCLE, COLOR_GREEN, -1)
            
            for point in transformed_points3:
                x,y = point
                cv2.circle(topview, (int(x),int(y)), BIG_CIRCLE, COLOR_GREEN, 2)
                cv2.circle(topview, (int(x),int(y)), SMALL_CIRCLE, COLOR_GREEN, -1)
            '''
            width = topview.shape[1]
            height= topview.shape[0]
            # Check if 2 or more people have been detected
            if len(transformed_points1) >= 2:
                
                # Iterate over every possible 2 by 2 between the points combinations 
                list_indexes = list(itertools.combinations(range(len(transformed_points1)), 2))
                for i,pair in enumerate(itertools.combinations(transformed_points1, r=2)):
                    # Check if the distance between each combination of points is less than the minimum distance chosen
                    if math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 ) < int(distance_minimum):
                        # Change the colors of the points that are too close from each other to red
                        if not (pair[0][0] > width or pair[0][0] < 0 or pair[0][1] > height+200  or pair[0][1] < 0 or pair[1][0] > width or pair[1][0] < 0 or pair[1][1] > height+200  or pair[1][1] < 0):
                            #change_color_on_topview(pair,topview)
                            index_pt1 = list_indexes[i][0]
                            index_pt2 = list_indexes[i][1]
                            cv2.rectangle(frame1, (results1[index_pt1][1][0],results1[index_pt1][1][1]), (results1[index_pt1][1][2],results1[index_pt1][1][3]), COLOR_BLUE, 2)
                            cv2.rectangle(frame1, (results1[index_pt2][1][0],results1[index_pt2][1][1]), (results1[index_pt2][1][2],results1[index_pt2][1][3]), COLOR_BLUE, 2)
                            violation_heatmap_points.append(pair[0]); violation_heatmap_points.append(pair[1])
                            
            #setting up gaussian kernel
            k = 21
            gauss = cv2.getGaussianKernel(k,np.sqrt(64))
            gauss = gauss*gauss.T
            gauss = (gauss/gauss[int(k/2),int(k/2)])                
            #blank image for heatmap points initialization
            img2 = np.zeros((topview.shape[0],topview.shape[1],3)).astype(np.float32)
            j = cv2.applyColorMap(((gauss)*255).astype(np.uint8),cv2.COLORMAP_JET).astype(np.float32)/255
            violation_heatmap_points_k = violation_heatmap_points[-skip:]
            
            for p in violation_heatmap_points_k:
                
                if(p[0]-int(k/2)>=0 and p[1]-int(k/2)>=0 and (p[0]+int(k/2))<topview.shape[0] and (p[1]+int(k/2))<topview.shape[1]):
                    
                    p[0] = p[0].astype(int); p[1] = p[1].astype(int)
                    b = img2[p[0]-int(k/2):(p[0]+int(k/2)+1)%topview.shape[0],p[1]-int(k/2):(p[1]+int(k/2)+1)%topview.shape[1],:]
                    c = j+b
                    img2[p[0]-int(k/2):(p[0]+int(k/2)+1)%topview.shape[0],p[1]-int(k/2):(p[1]+int(k/2)+1)%topview.shape[1],:]=c
            
            
            #inverse mask
            g = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            mask1 = np.where(g>0,1,0).astype(np.float32)
            mask_3 = np.ones((topview.shape[0],topview.shape[1],3))*(1-mask1)[:,:,None]
            mask_3 = mask_3/255
            mask = np.where(g>0,255,0).astype(np.float32)
            #multiply by mask
            mask_4=img2*(mask)[:,:,None]
            #generate new_topview
            new_topview = mask_3*topview
            #create heatmap
            heatmap = new_topview + mask_4/255

            
            
            #topview = cv2.resize(topview,(720,480))
            frame1 = cv2.resize(frame1,(720,480))
            #frame2 = cv2.resize(frame2,(720,480))
            #frame3 = cv2.resize(frame3,(720,480))
            #cv2.imshow(Name, topview)
            cv2.imshow(Name,heatmap)
            cv2.imshow("1", frame1)
            cv2.imshow("2", frame2)
            cv2.imshow("3", frame3)
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
        cam1.release()
        cam2.release()
        cam3.release()
        cv2.destroyWindow(Name)
        cv2.destroyWindow("1")
        cv2.destroyWindow("2")
        cv2.destroyWindow("3")        

    elif option == 10:
        
        video_path1 = "trim_1_vid_1.avi"
        points = set_points_from_mouse(video_path1,"1_last_frame.jpg",1080)
        

        top_view = "Topview_acad.png"
        points_t = set_points_from_mouse_topview(top_view,"1_last_frame.jpg")


    else:
        print("Invalid option entered. Exiting...")


main()
