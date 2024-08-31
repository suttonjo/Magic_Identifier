import sys
import cv2
import imutils
import numpy as np

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


def getQuad(grayscale, output):
    # Get the four corners of the card using contours and Hough line transform
    convex_hull_mask = np.zeros_like(grayscale)
    
    contours, _ = cv2.findContours(grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

    hull = [cv2.convexHull(contours[0])]
    cv2.drawContours(convex_hull_mask, hull, -1, 255, 1)
    cv2.imshow("Hull Mask", convex_hull_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    lines = cv2.HoughLinesP(convex_hull_mask, 1, np.pi / 200, 50, minLineLength=60, maxLineGap=2)
    sys.stdout.write("Lines: %s" % len(lines))

    if lines is not None and len(lines) == 4:
        params = [calcParams((line[0][0], line[0][1]), (line[0][2], line[0][3])) for line in lines]

        corners = []
        for i in range(len(params)):
            for j in range(i+1, len(params)):  # j starts at i+1 to avoid duplicate intersections
                intersec = FindIntersect(params[i], params[j])
                if intersec:
                    x, y = intersec
                    if 0 <= x < grayscale.shape[1] and 0 <= y < grayscale.shape[0]:
                        corners.append(intersec)
        
        if len(corners) == 4:
            for corner in corners:
                cv2.circle(output, corner, 3, (0, 0, 255), -1)
            return corners
    return []


def orderPoints(pts):
    # Rotate Image
    return [pts[3], pts[2], pts[1], pts[0]]


def main():
    image = cv2.imread("InputImages/Flowerfoot.jpeg")
    image = imutils.resize(image, height=700)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blurred_gray, 30, 100)

    card_corners = getQuad(edges, image)
    warped_card = np.zeros((400, 300, 3), dtype=np.uint8)
    
    if len(card_corners) == 4:
        ordered_corners = orderPoints(card_corners)
        card_corners_np = np.array(ordered_corners, dtype=np.float32)
        output_corners = np.array([[warped_card.shape[1], 0], [warped_card.shape[1], warped_card.shape[0]], [0, 0], [0, warped_card.shape[0]]], dtype=np.float32)
        homography, _ = cv2.findHomography(card_corners_np, output_corners)
        warped_card = cv2.warpPerspective(image, homography, (warped_card.shape[1], warped_card.shape[0]))

    cv2.imshow("Warped Card", warped_card)
    cv2.imshow("Edges", edges)
    cv2.imshow("Input", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()