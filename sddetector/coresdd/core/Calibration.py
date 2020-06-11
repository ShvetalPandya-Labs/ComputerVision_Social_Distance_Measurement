
import os
import argparse
import numpy as np
import cv2 as cv

from coresdd.common import CommonFunctionalities, CommonVideoUtils


# Workspace
WORKSPACE = os.path.dirname(os.path.realpath(__file__))

im_temp = np.zeros((0, 0, 3), np.uint8)
reference_pts = np.empty((0, 2), dtype=np.int32)


def get_video_input():
    # TODO : use following commented section to parse video input from command line
    """
    # USAGE : python3 Calibration.py --input ../data/TownCentreXVID.avi
    # parser = argparse.ArgumentParser(description='Use this script to run calibration of Video sequence to run Social '
    #                                              'Distance Detector using OpenCV.')
    # parser.add_argument('--input',
    #                     help='Path to input image or video file. Skip this argument to capture frames from a camera.')
    # args = parser.parse_args()
    # # Open a video file or an image file or a camera stream
    # video_input = args.input if args.input else 0
    """

    video_input = os.path.join('../../data/', 'TownCentreXVID.avi')
    slow_video_object = CommonVideoUtils.SlowVideoStream(video_input)
    frame_count = 0

    while True:
        frame = slow_video_object.video_stream_read(resize=True, width=1000, display_text=False)
        if frame_count == 0:
            if len(frame) > 0 and (frame.shape[0] > 0) and (frame.shape[1] > 0):

                # TODO : use following line to manually select four points from 1st Frame of the Video Stream
                #perspective_image_points = select_four_points_from_image(frame)

                perspective_image_points = np.asarray([[569, 116], [803, 146], [618, 332], [306, 279]], dtype="float32")
                #perspective_image_points = np.asarray([[567, 115], [801, 147], [491, 482], [145, 381]], dtype="float32")

                destination_image_points, ractangle_max_width, ractangle_max_height = \
                    get_destination_points(perspective_image_points, frame)

                # compute the perspective transform matrix and then apply it
                transform_matrix, status = cv.findHomography(perspective_image_points, destination_image_points)
                warped_perspective = cv.warpPerspective(frame, transform_matrix, (ractangle_max_width,
                                                                                  ractangle_max_height))
                # show the original and warped images
                #cv.imshow("Original", frame)
                #cv.imshow("Warped_small", warped_perspective)

                max_width, max_height, ref_point = \
                    get_destination_image_size(frame.shape[1], frame.shape[0], transform_matrix)

                destination_image_points = np.array([
                    [ref_point[0] + 0, ref_point[1] + 0],
                    [ref_point[0] + ractangle_max_width - 1, ref_point[1] + 0],
                    [ref_point[0] + ractangle_max_width - 1, ref_point[1] + ractangle_max_height - 1],
                    [ref_point[0] + 0, ref_point[1] + ractangle_max_height - 1]], dtype="float32")

                # compute the perspective transform matrix and then apply it
                transform_matrix, status = cv.findHomography(perspective_image_points, destination_image_points)
                warped_perspective = cv.warpPerspective(frame, transform_matrix, (max_width, max_height))
                # Draw grid on 1st Frame
                CommonFunctionalities.draw_grid(warped_perspective, line_color=(0, 255, 0), thickness=1,
                                                type=cv.LINE_AA, pxstep=50)

                # TODO : May be need to resize the image according to Aspect ratio
                #cv.imshow("Original", frame)
                #cv.imshow("Warped", warped_perspective)
                #cv.waitKey(0)
                break

        frame_count += 1

    return transform_matrix, max_width, max_height


def get_point_perspective_transform(x, y, transform_matrix):
    m_11 = transform_matrix[0][0]
    m_12 = transform_matrix[0][1]
    m_13 = transform_matrix[0][2]
    m_21 = transform_matrix[1][0]
    m_22 = transform_matrix[1][1]
    m_23 = transform_matrix[1][2]
    m_31 = transform_matrix[2][0]
    m_32 = transform_matrix[2][1]
    m_33 = transform_matrix[2][2]
    dest_x = int((m_11 * x + m_12 * y + m_13) / (m_31 * x + m_32 * y + m_33))
    dest_y = int((m_21 * x + m_22 * y + m_23) / (m_31 * x + m_32 * y + m_33))

    return [dest_x, dest_y]


def get_destination_image_size(source_image_width, source_image_height, transform_matrix):

    upper_left = get_point_perspective_transform(0, 0, transform_matrix)
    upper_right = get_point_perspective_transform(source_image_width, 0, transform_matrix)
    lower_right = get_point_perspective_transform(source_image_width, source_image_height, transform_matrix)
    lower_left = get_point_perspective_transform(0, source_image_height, transform_matrix)

    max_width = max(abs(upper_left[0] - upper_right[0]), abs(upper_left[0] - lower_left[0]),
                    abs(upper_left[0] - lower_right[0]),
                    abs(upper_right[0] - lower_right[0]), abs(upper_right[0] - lower_left[0]),
                    abs(lower_left[0] - lower_right[0]))

    max_height = max(abs(upper_left[1] - upper_right[1]), abs(upper_left[1] - lower_left[1]),
                    abs(upper_left[1] - lower_right[1]),
                    abs(upper_right[1] - lower_right[1]), abs(upper_right[1] - lower_left[1]),
                    abs(lower_left[1] - lower_right[1]))

    # Find left most, right most, top most and bottom most points
    left_most_value = min(upper_left[0], lower_left[0])
    top_most_value = min(upper_left[1], upper_right[1])

    # create destination_image_ref_point
    ref_point = [abs(left_most_value),  abs(top_most_value)]

    return max_width, max_height, ref_point


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def get_destination_points(perspective_image_points, source_image):
    # TODO : Use order_points fucntion afterwards
    tl = perspective_image_points[0]  # top left point
    tr = perspective_image_points[1]  # top right point
    br = perspective_image_points[2]  # bottom right point
    bl = perspective_image_points[3]  # bottom left point

    # Compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left
    # x-coordinates OR the top-right and top-left x-coordinates
    width_A = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    width_B = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    ractangle_max_width = max(int(width_A), int(width_B))

    # Compute the height of the new image, which will be the maximum distance between top-right and bottom-right
    # y-coordinates OR the top-left and bottom-left y-coordinates
    height_A = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    height_B = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    ractangle_max_height = max(int(height_A), int(height_B))

    # now that we have the dimensions of the new image, construct the set of destination points to obtain a
    # "birds eye view", (i.e. top-down view) of the image, again specifying points in the top-left, top-right,
    # bottom-right, and bottom-left order

    destination_image_points = np.array([
        [0, 0],
        [ractangle_max_width - 1, 0],
        [ractangle_max_width - 1, ractangle_max_height - 1],
        [0, ractangle_max_height - 1]], dtype="float32")

    return destination_image_points, ractangle_max_width, ractangle_max_height


def mouse_handler(event, x, y, flags, param):
    global im_temp, reference_pts

    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(im_temp, (x, y), 3, (0, 255, 255), 5, cv.LINE_AA)
        cv.imshow("Image", im_temp)
        if len(reference_pts) < 4:
            reference_pts = np.append(reference_pts, [(x, y)], axis=0)


def select_four_points_from_image(image):
    global im_temp, reference_pts

    # Create a window
    cv.namedWindow("Image", 1)
    cv.setMouseCallback("Image", mouse_handler)

    clone = image.copy()
    im_temp = image
    reference_pts = np.empty((0, 2), dtype=np.int32)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv.imshow("Image", image)
        key = cv.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    cv.destroyAllWindows()
    return reference_pts.copy()


if __name__ == '__main__':
    get_video_input()
