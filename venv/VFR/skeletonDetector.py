import cv2
import time
import numpy as np
from PIL import Image


# COCO Dataset -----------------------------
protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel"
nPoints = 18
POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
              [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]


net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

def process(frame, inWidth, inHeight, threshold):
    global net

    t = time.time()

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    #TODO: blob google it
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):

        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8,
                (255, 50, 0), 2, lineType=cv2.LINE_AA)

    return frame, points

img = cv2.imread('coat.png')
input_img, input_points = process(img.copy(), 100, 110, 0.05) # image 800 --> 150 | image 680 ---> 100

cv2.imshow("ff", input_img)
cv2.waitKey()

def skeleton(frame):

    frame, frame_points = process(frame, 168, 200, 0.3)

    frame_pnts = [frame_points[2], frame_points[5], frame_points[8], frame_points[11]]
    input_pnts = [input_points[2], input_points[5], input_points[8], input_points[11]]

    if None not in frame_pnts:
        input_pnts = np.float32([kp for kp in input_pnts])
        frame_pnts = np.float32([kp for kp in frame_pnts])

        frame_rows, frame_cols, _ = frame.shape

        M = cv2.getPerspectiveTransform(input_pnts, frame_pnts)
        dst = cv2.warpPerspective(img, M, (frame_cols,frame_rows))

        overlay = cv2.add(frame, dst)
        # overlay = cv2.addWeighted(frame, 1, dst, 1, 0.5)

        return overlay

    return frame
