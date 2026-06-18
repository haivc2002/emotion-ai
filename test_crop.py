import cv2
from mtcnn import MTCNN
import sys

img_path = sys.argv[1]
img = cv2.imread(img_path)
detector = MTCNN()
results = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print(results)
