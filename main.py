import pyautogui
import math
import cv2
import numpy as np
from ultralytics  import YOLO

# So we know who is jungling
# selectedChampion = input("What Champion should I be looking out for?\n")

def whichLane(source,labelFilePath):
    fileLines = open(labelFilePath, "r")
    champInfo = fileLines.readlines()[-1].split()
    xc, yc= float(champInfo[1]), float(champInfo[2])

    img_height, img_width = source.shape[0], source.shape[1]
    x_center, y_center = xc * img_width, yc * img_height

    championPosition = (x_center, y_center)
    
    # rather than using dots draw triangles and mark zones that are considered Top, Mid, or Bot lane. 
    # as soon as enemy enters that region/triangle's area, display the text 

    # Mid Lane plots
    cv2.circle(source, center= (int(img_width/2),int(img_height/2)), radius=6, color=(0,255,0), thickness=-1)
    cv2.circle(source, center= (int(img_width/2) - 100, int(img_height/2) + 100), radius=6, color=(0,255,0), thickness=-1)
    cv2.circle(source, center= (int(img_width/2) + 100,int(img_height/2) - 100), radius=6, color=(0,255,0), thickness=-1)
    cv2.circle(source, center= (int(img_width/2) - 50, int(img_height/2) + 50), radius=6, color=(0,255,0), thickness=-1)
    cv2.circle(source, center= (int(img_width/2) + 50,int(img_height/2) - 50), radius=6, color=(0,255,0), thickness=-1)
     
    # Top Lane plots
    cv2.circle(source, center= (int(img_width/2) - 100, int(img_height/2) - 100), radius=6, color=(255,0,0), thickness=-1)
    cv2.circle(source, center= (int(img_width/2) - 100, int(img_height/2) + 50), radius=6, color=(255,0,0), thickness=-1)
    cv2.circle(source, center= (int(img_width/2) - 100, int(img_height/2) - 20), radius=6, color=(255,0,0), thickness=-1)
    cv2.circle(source, center= (int(img_width/2) + 50, int(img_height/2) - 100), radius=6, color=(255,0,0), thickness=-1)
    cv2.circle(source, center= (int(img_width/2) - 20, int(img_height/2) - 100), radius=6, color=(255,0,0), thickness=-1)
    
    # Bot Lane plots
    cv2.circle(source, center= (int(img_width/2) + 100, int(img_height/2) + 100), radius=6, color=(0,0,255), thickness=-1)
    cv2.circle(source, center= (int(img_width/2) + 100, int(img_height/2) - 50), radius=6, color=(0,0,255), thickness=-1)
    cv2.circle(source, center= (int(img_width/2) + 100, int(img_height/2) + 20), radius=6, color=(0,0,255), thickness=-1)
    cv2.circle(source, center= (int(img_width/2) - 50, int(img_height/2) + 100), radius=6, color=(0,0,255), thickness=-1)
    cv2.circle(source, center= (int(img_width/2) + 20, int(img_height/2) + 100), radius=6, color=(0,0,255), thickness=-1)
    
    cv2.imshow("Lanes", source)

    return source
    

def draw_rectangle(event,x,y,flags,param):

    global pt1,pt2,topLeft_clicked,botRight_clicked

    # get mouse click
    if event == cv2.EVENT_LBUTTONDOWN:

        if topLeft_clicked == True and botRight_clicked == True:
            topLeft_clicked = False
            botRight_clicked = False
            pt1 = (0,0)
            pt2 = (0,0)

        if topLeft_clicked == False:
            pt1 = (x,y)
            topLeft_clicked = True
            
        elif botRight_clicked == False:
            pt2 = (x,y)
            botRight_clicked = True


cv2.namedWindow("Live")
cv2.setMouseCallback('Live', draw_rectangle) 
cv2.resizeWindow("Live", 1920, 1080)

pt1 = (0,0)
pt2 = (0,0)
topLeft_clicked = False
botRight_clicked = False            

model = YOLO("C:/Users/baosh/LoLMapVision/trained/weights/best.pt")

while True:
    img = pyautogui.screenshot()
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if topLeft_clicked:
        cv2.circle(frame, center=pt1, radius=5, color=(0,0,255), thickness=-1)
    
    if topLeft_clicked and botRight_clicked:
        cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
        y1, x1 = pt1
        y2, x2 = pt2

        mapRegion = pyautogui.screenshot(region=(y1, x1, y2-y1, x2-x1))
        Map = np.array(mapRegion)
        Map = cv2.cvtColor(Map, cv2.COLOR_BGR2RGB)
        
        # I want to filter by "classes" with the selectedChampion input from the start
        model.predict(Map, save=True, save_txt=True, save_conf=True)
        predictfilepath = "C:/Users/baosh/LoLMapVision/runs/detect/predict/labels/image0.txt"
        lane = whichLane(Map,predictfilepath)
        
        # if lane == "fog":
        #     cv2.putText(Map, text="Fog",org=(0,0),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=10, color=(255,0,0), thickness=4)
        # elif lane == "mid":
        #     cv2.putText(Map, text="MidLane",org=(0,0),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=10, color=(255,0,0), thickness=4)
        # elif lane == "top":
        #     cv2.putText(Map, text="TopLane",org=(0,0),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=10, color=(255,0,0), thickness=4)
        # elif lane == "bot":
        #     cv2.putText(Map, text="BotLane",org=(0,0),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=10, color=(255,0,0), thickness=4)
        
        cv2.imshow("Map", Map)

    model.predict(frame, save=True, save_txt=True, save_conf=True)
    cv2.imshow('Live', frame)
        
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()





