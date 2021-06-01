#Niko Severino
#COM 496
#OpenPose hand landmarks extraction script

#Extracts hand landmarks and organizes data into
#text files to be fed into LSTM 

#-----------------------------------------------------#
#Dependencies 
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import time 
import math


try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))

    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../x64/Release;' +  dir_path + '/../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
   
    params = dict()
    params["model_folder"] = "../../models/"
    params["hand"] = True
    params["hand_detector"] = 3
    params["net_resolution"] = '320x320'
    params["number_people_max"] = 1     #To avoid catching shadows in some clips

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    #Initialize class_frames.txt variables
    clip_count = 0
    frame_count = 0

    #loop through class files
    for k in range(6):
        if(k==0):
            route = r"C:\MCNE\data\pitcher"
            gest_class = 1 
        elif(k==1):
            route = r"C:\MCNE\data\beer"
            gest_class = 2
        elif(k==2):
            route = r"C:\MCNE\data\shot"
            gest_class = 3
        elif(k==3):
            route = r"C:\MCNE\data\one"
            gest_class = 4
        elif(k==4):
            route = r"C:\MCNE\data\two"
            gest_class = 5
        elif(k==5):
            route = r"C:\MCNE\data\three"
            gest_class = 6
          
        #lists directory contents of media and appends their path to a list
        videoPaths = []
        for path in os.listdir(route):
            full_path = os.path.join(route, path)
            if os.path.isfile(full_path):
                videoPaths.append(full_path)
                
        print(videoPaths)
    
        #acquire frame from video 
        for gesture in videoPaths:
            
            #Tracks frames and gesture class
            output = open(r"C:\MCNE\data\OP_landmarks\data\temp_load\Class_Frames.txt","a")
            output.write(str(clip_count) + "\t" + str(gest_class) + "\t" + str(frame_count + 1) + "\t") 
            
            #Tracks the clip count
            clip_count = clip_count + 1 

            #Video capture
            datum = op.Datum()
            cap = cv2.VideoCapture(gesture)
            fps = cap.get(cv2.CAP_PROP_FPS)
        
            video = None

            #Temp frame count
            temp_count = 0

            #Open video frames
            while cap.isOpened():
                
                #read frame
                ret,frame = cap.read()
                
                #if there is a frame
                if ret == True:

                    temp_count = temp_count + 1
                    frame_count = frame_count + 1
                    
                    #resize frame to keep consistent
                    frame = cv2.resize(frame,(854,480))

                    #Detect hands/body in frame
                    datum.cvInputData = frame
                    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                    opframe=datum.cvOutputData      

                    #Output left and right keypoints
                    print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
                    print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))

                    cv2.imshow("Acquired Hand Keypoints", datum.cvOutputData)

                    #Raw data (frame + unnormalized keypoints).
                    File_output = open(r"C:\MCNE\data\OP_landmarks\data\temp_load\Face_data.txt","a")
                    
                    #Normalized data 
                    File_output2 = open(r"C:\MCNE\data\OP_landmarks\data\temp_load\Normalized_data.txt","a")

                    #Initalize value variables
                    raw_coords = ""
                    normalized_coords = ""
                    counter = 0

                    #Start hand data extraction/organization
                    #20 hand keypoints
                    #-----------------------------------------#
                    for i in range(21):

                        #For raw data
                        raw_temp = ""
                     
                        norm_temp = "" 
                 
                        #First loop hands

                        #Neck point - Hip point
                        scale =  math.floor(datum.poseKeypoints[0][8][1] - datum.poseKeypoints[0][1][1])
                        
                        #Raw Loop

                        #For each hand
                        for hand in range(2):
                            #Second loop for coords 
                            for j in range(2):
                           
                                #[hand] [array] [keypoint x y c] [x and y pair] 
                                raw_temp += (str(datum.handKeypoints[hand][0][i][j].round(2)) + " ")

                        #Normalized loop        
                        for j in range(2):
                            #Subtract from wrists for hands, divide by length of body to normalize/scale
                            #0 = left hand, 1 = right hand
                            norm_temp += (str(((datum.handKeypoints[0][0][i][j] - datum.poseKeypoints[0][4][j])/scale).round(2)) + " "  +
                                (str(((datum.handKeypoints[1][0][i][j] - datum.poseKeypoints[0][7][j])/scale).round(2)) + " "))

                        normalized_coords += str(norm_temp)  
                        raw_coords += raw_temp 
                    #--------------------------------------------#

                    #Begin face cropping/openpose 

                  
                    #Write to raw data file
                    File_output.write(str(gest_class) + "\t")
                    File_output.write(str(clip_count) + "\t")
                    File_output.write(str(temp_count) + "\t")#frame
                    File_output.write(str(raw_coords) + "\n")#coordinates
                    

                    #Write to normalized data file
                    File_output2.write(str(gest_class) + '\t' + str(clip_count) + '\t' + str(temp_count) + '\t' + str(normalized_coords) + "\n")
                    
                    
                    File_output.close()
                    File_output2.close()
                    #use matlab for visualization

                    cv2.waitKey(1)
                else:
                    break
            output.write(str(frame_count) + "\n")
        
            cv2.destroyAllWindows()  
            #video.release()
except Exception as e:
    print(e)
    sys.exit(-1)


