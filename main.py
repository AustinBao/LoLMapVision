import os
import pyautogui
import cv2
import numpy as np
from ultralytics  import YOLO

# Specify resolution
resolution = (1920, 1080)

codec = cv2.VideoWriter_fourcc(*"XVID")
filename = "Recording.avi"
fps = 60.0
out = cv2.VideoWriter(filename, codec, fps, resolution)

cv2.namedWindow("Live", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Live", 480, 270)

while True:
	# Take screenshot using PyAutoGUI
	img = pyautogui.screenshot()

	frame = np.array(img)
	
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Write it to the output file
	out.write(frame)
	
	cv2.imshow('Live', frame)
	
	if cv2.waitKey(1) == 27:
		break

out.release()
cv2.destroyAllWindows()




os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    model.train(data="C:/Users/baosh/LoLMapVision/MapDataset/data.yaml", epochs=150)