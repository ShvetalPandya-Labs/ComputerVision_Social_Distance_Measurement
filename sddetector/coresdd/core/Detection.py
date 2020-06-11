import os
import cv2 as cv

from coresdd.common import CommonVideoUtils, CommonFunctionalities
from coresdd.core import Calibration

from reference_codes.object_detection import FirstObjectDetection

from imageai.Detection import VideoObjectDetection

# TODO : NOTE this file is in rough development stage so this won't be final structure and code will much of rough.

transform_matrix_global = []
max_width_global = 0
max_height_global = 0


def video_object_detection_from_video_file():
    model_path = os.path.join('../../models/')
    video_file_input_path = os.path.join('../../data/')
    video_file_op_path = os.path.join('../../data/')

    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(model_path, "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel(detection_speed="fast")  # "normal"(default), "fast", "faster" , "fastest" and "flash"

    custom_objects = detector.CustomObjects(person=True, bicycle=True, motorcycle=True)

    video_path = detector.detectCustomObjectsFromVideo(
                    custom_objects=custom_objects,
                    input_file_path=os.path.join(video_file_input_path, 'TownCentreXVID.avi'),
                    output_file_path=os.path.join(video_file_op_path, 'TownCentreXVID_Detection'),
                    frames_per_second=2, per_second_function=for_seconds,
                    log_progress=True, return_detected_frame=True)

    print(video_path)


def for_frame(frame_number, output_array, output_count):
    print("FOR FRAME ", frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------ END OF A FRAME --------------")


def for_seconds(second_number, output_arrays, count_arrays, average_output_count, detected_copy):
    #print('shape of detected_copy : ', detected_copy.shape)
    warped_perspective = cv.warpPerspective(detected_copy, transform_matrix, (max_width, max_height))

    CommonFunctionalities.draw_grid(warped_perspective, line_color=(0, 255, 0), thickness=1,
                                    type=cv.LINE_AA, pxstep=50)

    # TODO : May be need to resize the image according to Aspect ratio
    cv.imshow("Original", detected_copy)
    cv.imshow("Warped_detected", warped_perspective)
    #cv.waitKey(0)
    #break

    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------ END OF A SECOND --------------")


def for_minute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)
    print("------------ END OF A MINUTE --------------")


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
                print(frame.shape)

        frame_count += 1
        print('frame_count :', frame_count)
        if (frame_count % 500) == 0:
            detected_copy, output_objects_array = FirstObjectDetection.object_detection_from_image()

            cv.imshow('Detected', detected_copy)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            print('output_objects_array : ', output_objects_array)


if __name__ == '__main__':
    #get_video_input()
    transform_matrix, max_width, max_height = Calibration.get_video_input()
    transform_matrix_global = transform_matrix
    max_width_global = max_width
    max_height_global = max_height
    print('transform_matrix_global : ', transform_matrix_global)
    print('max_width_global : ', max_width_global)
    print('max_height_global : ', max_height_global)
    video_object_detection_from_video_file()
