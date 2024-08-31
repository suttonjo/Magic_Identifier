from skimage import exposure
import numpy as np
import imutils
import cv2

# Load the image and check if it's loaded correctly
image = cv2.imread("InputImages/Flowerfoot.jpeg")

if image is None:
    print("Error: Image not loaded. Please check the file path.")
else:
    # Resize image and keep original for later use
    ratio = image.shape[0] / 700.0
    orig = image.copy()
    image = imutils.resize(image, height=700)

    # Convert to grayscale, apply blur, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)

    # Find contours and select the largest ones
    cnts = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    # Look for a contour that approximates a rectangle
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        
        if len(approx) == 4:
            screenCnt = c
            break

# Get the minimum area rectangle and draw it
rect = cv2.minAreaRect(screenCnt)
box = cv2.boxPoints(rect)
box = np.int32(box)

cv2.drawContours(image, [box], 0, (0, 255, 0), 3)
cv2.imshow("Card with MinAreaRect", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Reorder points for perspective transform
pts = box.reshape(4, 2)
rect = np.zeros((4, 2), dtype="float32")

# Top-left has the smallest sum, bottom-right has the largest
s = pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]

# Top-right has the smallest difference, bottom-left has the largest
diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]

# Scale rectangle to original image size
rect *= ratio

# Calculate width and height of the new image
(tl, tr, br, bl) = rect
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

# Set maximum width and height
maxWidth = max(int(widthA), int(widthB))
maxHeight = max(int(heightA), int(heightB))

# Define destination points for perspective transform
dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype="float32")

# Apply perspective transform to get a top-down view
M = cv2.getPerspectiveTransform(rect, dst)
warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

# Save the cropped image and display results
cv2.imshow("warp", imutils.resize(warp, height=700))
cv2.waitKey(0)
cv2.destroyAllWindows()