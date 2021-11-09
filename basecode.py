import cv2
import threading

#creating a class of camera to create camera objects in separate threads
class camThread(threading.Thread):
    def __init__(self, Name, IPAddress, recording):
        threading.Thread.__init__(self)
        self.Name = Name
        self.IPAddress = IPAddress
        self.recording = recording
    def run(self):
        print("Starting " + self.Name)
        camProcess(self.Name, self.IPAddress, self.recording)


def camProcess(Name, IPAddress, recording):
    if recording==0:   #if recording mode is set off, then only stream
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
        cv2.destroyWindow(Name)
    else:				#if recording mode is set on, then record the stream
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
        outputFile.release()
        cv2.destroyWindow(Name)

def main():

    print("Press 1 to record stream, 2 for live stream: ")
    option = int(input())

    if option == 1:
        # Record video
        thread1 = camThread("Stream1Recording", "http://10.130.21.153:8080/video",1)
        thread2 = camThread("Stream2Recording", "http://10.130.21.174:8080/video",1)
        thread3 = camThread("Stream3Recording", "http://10.130.146.242:8080/video",1)

        thread1.start()
        thread2.start()
        thread3.start()
        print()
        print("Number of current threads: ", threading.activeCount())


    elif option == 2:
        # live stream
        thread1 = camThread("Live Stream of Camera 1", "http://10.130.21.153:8080/video",0)
        thread2 = camThread("Live Stream of Camera 2", "http://10.130.21.174:8080/video",0)
        thread3 = camThread("Live Stream of Camera 3", "http://10.130.146.242:8080/video",0)

        thread1.start()
        thread2.start()
        thread3.start()
        print()
        print("Number of current threads: ", threading.activeCount())

    else:
        print("Invalid option entered. Exiting...")


main()
