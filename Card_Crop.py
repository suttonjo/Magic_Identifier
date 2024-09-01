import sys
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
import os
import math

def calcParams(p1, p2):
    # Calculate line parameters a, b, c for the line equation ax + by + c = 0
    if p2[1] - p1[1] == 0:  # Horizontal line
        a = 0.0
        b = -1.0
    elif p2[0] - p1[0] == 0:  # Vertical line
        a = -1.0
        b = 0.0
    else:  # General case
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = -1.0
    c = -a * p1[0] - b * p1[1]
    return np.array([a, b, c])


def FindIntersect(params1, params2):
    # Calculate intersection of two lines given by params1 and params2
    det = params1[0] * params2[1] - params2[0] * params1[1]
    if abs(det) < 0.5:  # Lines are approximately parallel
        return None
    else:
        x = (params2[1] * -params1[2] - params1[1] * -params2[2]) / det
        y = (params1[0] * -params2[2] - params2[0] * -params1[2]) / det
        return (int(x), int(y))


def getQuad(image):
    # modify image to get edges of picture
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (3,3),0)
    edges = cv2.Canny(blurred_gray, 30, 100)

    # Get the four corners of the card using contours and Hough line transform
    convex_hull_mask = np.zeros_like(edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

    hull = [cv2.convexHull(contours[0])]
    cv2.drawContours(convex_hull_mask, hull, 0, 255, 1)

    lines = cv2.HoughLines(convex_hull_mask, 1, np.pi / 230, 150, None, 0, 0)
    sys.stdout.write("Lines: %s" % len(lines))

    linePoints = []

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = [int(x0 + 1000*(-b)), int(y0 + 1000*(a))]
            pt2 = [int(x0 - 1000*(-b)), int(y0 - 1000*(a))]
            linePoints.append([pt1, pt2])



    
    [plt.plot([line[0][0],line[1][0]],[line[0][1],line[1][1]] , color="green", linewidth = 3) for line in linePoints]
    plt.imshow(image)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


    if lines is not None:
        params = [calcParams((line[0][0], line[0][1]), (line[1][0], line[1][1])) for line in linePoints]

        corners = []
        for i in range(len(params)):
            for j in range(i+1, len(params)):  # j starts at i+1 to avoid duplicate intersections
                intersec = FindIntersect(params[i], params[j])
                if intersec:
                    x, y = intersec
                    if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:
                        corners.append(intersec)

        """
        Do getting rid of duplicates and getting rid of inbetweens seperately
        Work on the case when you look at two opposite corners and how you can't remove something if thats the case
        (you want to be looking at two corners sharing an edge)

        threshold = 30
        cornersCopy = corners
        for corner in cornersCopy:
            cIndex = corner.index
            removed = False
            for corner1 in corners:
                c1Index = corner1.index
                if(cIndex != c1Index):
                    for corner2 in corners:
                        c2Index = corner2.index
                        if(cIndex != c2Index) and (c1Index != c2Index):
                            x = corner[0]
                            y = corner[1]

                            x1 = corner1[0]
                            y1 = corner1[1]

                            x2 = corner2[0]
                            y2 = corner2[1]

                            dist = abs(x1-x + y1-y)
                            if ((x1 <= x <= x2 or x2 <= x <= x1)
                                or (y1 <= y <= y2 or y2 <= y <= y1)):
                                corners.remove(corner)
                                removed = True
                                break
                            
                            if (dist < threshold):
                                corners.remove(corner)
                                removed = True
                                break
                    if removed:
                        break

        """

        if len(corners) == 4:
            for corner in corners:
                cv2.circle(image, corner, 3, (255, 0, 255), 0)
            return corners
    return []

def warpCard(image, length, width):
    # Find corners and perspective warp the card to find the card
    card_corners = getQuad(image)
    warped_card = np.zeros((length, width, 3), dtype=np.uint8)
    
    if len(card_corners) == 4:
        ordered_corners = orderPoints(card_corners)
        card_corners_np = np.array(ordered_corners, dtype=np.float32)
        output_corners = np.array([[warped_card.shape[1], 0], [warped_card.shape[1], warped_card.shape[0]], [0, 0], [0, warped_card.shape[0]]], dtype=np.float32)
        homography, _ = cv2.findHomography(card_corners_np, output_corners)
        warped_card = cv2.warpPerspective(image, homography, (warped_card.shape[1], warped_card.shape[0]))
        
        return warped_card
    return []

def orderPoints(pts):
    # Rotate Image
    # Convert points to a numpy array
    pts = np.array(pts)
    
    # Calculate the sum of the coordinates (x + y)
    s = pts.sum(axis=1)
    
    # Calculate the difference of the coordinates (y - x)
    diff = np.diff(pts, axis=1).flatten()
    
    # The top-left point will have the smallest sum
    top_left = pts[np.argmin(s)]
    
    # The bottom-right point will have the largest sum
    bottom_right = pts[np.argmax(s)]
    
    # The top-right point will have the smallest difference
    top_right = pts[np.argmin(diff)]
    
    # The bottom-left point will have the largest difference
    bottom_left = pts[np.argmax(diff)]
    
    return [top_right, bottom_right, top_left, bottom_left]

def getTitle(card, width, length):
    titleHeight = int(length * .1125)
    titleWidth = int(width * .882)

    outerEdge = int(length * .048)

    titleImage = card[
        outerEdge : titleHeight,
        outerEdge : outerEdge + titleWidth
    ]

    return titleImage


def main():
    path = "InputImages"
    dir_list = os.listdir(path)
    print("Files and directories in '", path, "' :")
    # prints all files
    print(dir_list)
    for imageName in dir_list:
        print(imageName)
        image = cv2.imread("InputImages/" + str(imageName))
        image = imutils.resize(image, height=1200)
        cv2.imshow("Input", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        width = 700
        length = 980

        
        warped_card = warpCard(image, length, width)

        cv2.imshow("Warped Card", warped_card)
        cv2.imshow("Input", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        title = getTitle(warped_card, width, length)

        cv2.imshow("Title", title)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



main()