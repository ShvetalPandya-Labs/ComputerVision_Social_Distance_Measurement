
import os
import cv2 as cv

from imageai.Detection import ObjectDetection


def object_detection_from_image():
    model_path = os.getcwd()
    input_image_path = os.getcwd()
    output_image_path = os.getcwd()

    detector = ObjectDetection()
    #detector.setModelTypeAsRetinaNet()
    #detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
    #detector.setModelTypeAsYOLOv3()
    #detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(os.path.join(model_path, "yolo-tiny.h5"))
    detector.loadModel()
    # detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "TownCentreImage.png"),
    #                                               output_image_path=os.path.join(execution_path, "image2new.jpg"),
    #                                               minimum_percentage_probability=30)

    detected_copy, output_objects_array = detector.detectCustomObjectsFromImage(
        custom_objects=None,
        input_image=os.path.join(input_image_path, "TownCentreImage.png"),
        output_image_path=os.path.join(output_image_path, "image2new.jpg"),
        input_type="file",
        output_type="array", extract_detected_objects=False,
        minimum_percentage_probability=50, display_percentage_probability=True,
        display_object_name=True, thread_safe=False)

    return detected_copy, output_objects_array




# for eachObject in detections:
#     print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
#     print("--------------------------------")