import cv2
import time
import numpy as np

# COCO Dataset -----------------------------
protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel"
nPoints = 18
POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
              [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

#
# inWidth = 368
# inHeight = 368

inWidth = 168
inHeight = 168

threshold = 0.3

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)

def process(frame):
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


    # ids = []
    # for pair in POSE_PAIRS:
    #     partA = pair[0]
    #     partB = pair[1]
    #
    #     ids.append(partA)
    #     ids.append(partB)
    #
    # srcMat = []
    # for pnt in points:
    #     dstMat.append(pnt)
    #
    # srcMat = np.array(srcMat)
    #
    # (srcH, srcW) = source.shape[:2]
    # srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8,
                (255, 50, 0), 2, lineType=cv2.LINE_AA)

    return frame, points


def skeleton(frame):
    img = cv2.imread('muscle-human-body.jpg')
    input_img, input_points = process(img)
    frame, frame_points = process(frame)


    input_points = np.float32([kp for kp in input_points])
    frame_points = np.float32([kp for kp in frame_points])

    if len(frame_points) == len(input_points):
        (H, _) = cv2.findHomography(frame_points, input_points, cv2.RANSAC)
        warped = cv2.warpPerspective(frame, H, (inWidth, inHeight))
        return warped

    return None
