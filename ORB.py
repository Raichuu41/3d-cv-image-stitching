import cv2
import numpy as np
import imutils


def extract_features(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors in the image
    kp, des = orb.detectAndCompute(gray, None)

    return kp, des


def match_features(kp1, des1, kp2, des2):
    # Match features between the images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Select the top matches
    num_good_matches = int(len(matches) * 0.90)
    matches = matches[:num_good_matches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    return points1, points2


def estimate_homography(points1, points2):
    # Find homography matrix
    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    return H


def warp_images(img1, img2, H):
    # Warp right image to align with left image
    height, width, channels = img1.shape
    img2_aligned = cv2.warpPerspective(img2, H, (width + img2.shape[1], height))
    img2_aligned[:height, :width, :] = img1

    return img2_aligned


def main():
    # Load left and right images
    img_left = cv2.imread('01.jpg')
    img_right = cv2.imread('02.jpg')

    # Extract features from both images
    kp1, des1 = extract_features(img_left)
    kp2, des2 = extract_features(img_right)

    # Match features between the images
    points1, points2 = match_features(kp1, des1, kp2, des2)

    # Estimate homography matrix
    H = estimate_homography(points1, points2)

    # Warp right image to align with left image
    img_aligned = warp_images(img_left, img_right, H)

    # Blend images
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch([img_left, img_right])

    # Show the result
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
