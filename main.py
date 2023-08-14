import pyautogui
import cv2
import numpy as np
from ultralytics  import YOLO

# So we know who is jungling
selectedChampion = input("What Champion should I be looking out for?\n")


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

model = YOLO("C:/OpenCV/LoLMapVision/runs/detect/train3/weights/best.pt")

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
        model.predict(Map, save=True, save_txt=True, save_conf=True, max_det=1)
        # redict_img_path = "C:/OpenCV/LoLMapVision/runs/detect/predict/image0.jpg"
        # new_img = cv2.imread(predict_img_path)

        cv2.imshow("Map", Map)

    
    cv2.imshow('Live', frame)
        
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()





