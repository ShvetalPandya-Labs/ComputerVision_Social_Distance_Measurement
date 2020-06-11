import cv2 as cv


def draw_grid(img, line_color=(0, 255, 0), thickness=1, type=cv.LINE_AA, pxstep=50):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type, thickness=thickness)
        y += pxstep


def draw_lines(image):
    # Window name in which image is displayed
    window_name = 'Lines Image'

    image.shape[0] / 100
    image.shape[1] / 100
    # Start coordinate, here (0, 0)
    # represents the top left corner of image
    start_point = (0, 0)

    # End coordinate, here (250, 250)
    # represents the bottom right corner of image
    end_point = (250, 250)

    # Green color in BGR
    color = (0, 255, 0)

    # Line thickness of 9 px
    thickness = 9

    # Using cv2.line() method
    # Draw a diagonal green line with thickness of 9 px
    image = cv.line(image, start_point, end_point, color, thickness)

    # Displaying the image
    cv.imshow(window_name, image)