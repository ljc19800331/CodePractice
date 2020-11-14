
'''
1. The review algorithms:
2. The robot slam problems -- pending

# The question is at the email -- pending

'''

class RobotMove():

    def __init__(self):
        self.a = 1

    def Move(self):

        a = 1

import cv2

img = cv2.imread("../raccoon.png")
print(img.shape)

import numpy as np

a = np.linalg.norm([1,2,3, 4])
print("a = ", a)