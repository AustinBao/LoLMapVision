import pyautogui
import math
import cv2
import os
import numpy as np
from ultralytics  import YOLO

# So we know who is jungling
selectedChampion = input("What Champion should I track?\n")

def delete_directory(directory_path):
    file_list = os.listdir(directory_path)
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            delete_directory(file_path)

    os.rmdir(directory_path)
    print(f"Directory '{directory_path}' and its contents successfully deleted.")

def whichChamp(champName):
    names = ['Aatrox', 'Ahri', 'Akali', 'Alistar', 'Amumu', 'Anivia', 'Annie', 'Aphelios', 'Ashe', 'AurelionSol', 'Azir', 'Bard', 'Blitzcrank', 'Brand', 'Braum', 'Caitlyn', 'Camille', 'Cassiopeia', 'Chogath', 'Corki', 'Darius', 'Diana', 'DrMundo', 'Draven', 'Ekko', 'Elise', 'Evelynn', 'Ezreal', 'Fiddlesticks', 'Fiora', 'Fizz', 'Galio', 'Gangplank', 'Garen', 'Gnar', 'Gragas', 'Graves', 'Gwen', 'Hecarim', 'Heimerdinger', 'Illaoi', 'Irelia', 'Ivern', 'Janna', 'Jarvan', 'Jax', 'Jayce', 'Jhin', 'Jinx', 'Kaisa', 'Kalista', 'Kane', 'Karma', 'Karthus', 'Kassadin', 'Katarina', 'Kayle', 'Kayn', 'Kennen', 'KhaZix', 'Kled', 'KogMaw', 'LeBlanc', 'LeeSin', 'Leona', 'Lillia', 'Lissandra', 'Lucian', 'Lulu', 'Lux', 'Malphite', 'Malzahar', 'Maokai', 'MasterYi', 'Mindred', 'MissFortune', 'Mordekaiser', 'Morgana', 'Nami', 'Nasus', 'Nautilus', 'Neeko', 'Nidalee', 'Nilah', 'Nocturne', 'Nunu', 'Olaf', 'Orianna', 'Ornn', 'Pantheon', 'Poppy', 'Pyke', 'Qiyana', 'Quinn', 'Rakan', 'Rammus', 'Reksai', 'Rell', 'Renata', 'Renekton', 'Rengar', 'Riven', 'Rumble', 'Ryze', 'Samira', 'Sejuani', 'Senna', 'Seraphine', 'Sett', 'Shaco', 'Shen', 'Shyvana', 'Singed', 'Sion', 'Sivir', 'Skarner', 'Sona', 'Soraka', 'Swain', 'Sylas', 'Syndra', 'TahmKench', 'Taliyah', 'Talon', 'Taric', 'Teemo', 'Thresh', 'Tristana', 'Trundle', 'Tryndamere', 'TwistedFate', 'Twitch', 'Udyr', 'Urgot', 'Varus', 'Vayne', 'Veigar', 'Velkoz', 'Vex', 'Vi', 'Viego', 'Viktor', 'Vladimir', 'Volibear', 'Warwick', 'Wukong', 'Xayah', 'Xerath', 'XinZhao', 'Yasuo', 'Yone', 'Yorick', 'Yuumi', 'Zac', 'Zed', 'Zeri', 'Ziggs', 'Zilean', 'Zoe', 'Zyra']
    return names.index(champName)

def whichLane(source,labelFilePath):
    if os.path.getsize(labelFilePath) == 0:
        return "No Detection"

    fileLines = open(labelFilePath, "r")    
    champInfo = fileLines.readlines()[-1].split()
    xc, yc= float(champInfo[1]), float(champInfo[2])
    img_height, img_width = source.shape[0], source.shape[1]
    x_champ, y_champ = xc * img_width, yc * img_height

    championPosition = (x_champ, y_champ)

    botLaneCords =[(int(img_width/2) + 100, int(img_height/2) + 100),
                   (int(img_width/2) + 100, int(img_height/2) - 50 ),
                   (int(img_width/2) + 100, int(img_height/2) + 20),
                   (int(img_width/2) - 50, int(img_height/2) + 100),
                   (int(img_width/2) + 20, int(img_height/2) + 100) ]
    
    topLaneCords =[(int(img_width/2) - 100, int(img_height/2) - 100),
                   (int(img_width/2) - 100, int(img_height/2) + 50),
                   (int(img_width/2) - 100, int(img_height/2) - 20),
                   (int(img_width/2) + 50, int(img_height/2) - 100),
                   (int(img_width/2) - 20, int(img_height/2) - 100) ]
    
    midLaneCords =[(int(img_width/2),int(img_height/2)),
                   (int(img_width/2) - 100, int(img_height/2) + 100),
                   (int(img_width/2) + 100,int(img_height/2) - 100),
                   (int(img_width/2) - 50, int(img_height/2) + 50),
                   (int(img_width/2) + 50,int(img_height/2) - 50) ]
    
    allLanes = [topLaneCords, midLaneCords, botLaneCords]

    topDistance = midDistance = botDistance = 1000
    allDistance = []

    for t in topLaneCords:
        cv2.circle(source, center=t , radius=6, color=(0,255,0), thickness=-1)
        champ_distance_from_circle = math.sqrt( (abs(championPosition[0]-t[0])**2) + (abs(championPosition[1]-t[1])**2) )
        if champ_distance_from_circle < topDistance:
            topDistance = champ_distance_from_circle         
    for m in midLaneCords:
        cv2.circle(source, center=m , radius=6, color=(0,0,255), thickness=-1)
        champ_distance_from_circle = math.sqrt( (abs(championPosition[0]-m[0])**2) + (abs(championPosition[1]-m[1])**2) )
        if champ_distance_from_circle < midDistance:
            midDistance = champ_distance_from_circle    
    for b in botLaneCords:
        cv2.circle(source, center=b , radius=6, color=(255,0,0), thickness=-1)
        champ_distance_from_circle = math.sqrt( (abs(championPosition[0]-b[0])**2) + (abs(championPosition[1]-b[1])**2) )
        if champ_distance_from_circle < botDistance:
            botDistance = champ_distance_from_circle 

    allDistance.append(topDistance)
    allDistance.append(midDistance)
    allDistance.append(botDistance)

    if allDistance[0] == 1000 and allDistance[1] == 1000 and allDistance[2] == 1000:
        return "Fog"
    else:
        closest_lane = allDistance.index(min(allDistance))    
        if closest_lane == 0:
            return "Top"
        elif closest_lane == 1:
            return "Mid"
        elif closest_lane == 2:
            return "Bot"


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

#"C:/OpenCV/LoLMapVision/trained/weights/best.pt"
#"C:/Users/baosh/LoLMapVision/trained/weights\best.pt"
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
        
        champInt = int(whichChamp(selectedChampion))
        model.predict(Map, save=True, save_txt=True, save_conf=True, classes=champInt)

        #"C:/OpenCV/LoLMapVision/runs/detect/predict11/labels/image0.txt"
        # C:\Users\baosh\LoLMapVision\runs\detect
        closest_lane = whichLane(Map, "C:/Users/baosh/LoLMapVision/runs/detect/predict/labels/image0.txt")
        cv2.putText(Map,text=f"{closest_lane}",org=(10,int(Map.shape[1] - 100)), fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,
                    color=(255,255,255),thickness=2)

        cv2.imshow("Map", Map)


    cv2.imshow('Live', frame)
        
    if cv2.waitKey(1) == 27:
        delete_directory("C:/Users/baosh/LoLMapVision/runs/detect/predict")
        break

cv2.destroyAllWindows()

