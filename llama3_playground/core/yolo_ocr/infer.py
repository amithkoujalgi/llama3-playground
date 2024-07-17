import cv2
from ultralytics import YOLO


def infer_from_model(src_image_file: str, target_image_file: str, model_dir: str):
    MODEL = YOLO(f"{model_dir}/weights/best.pt")  # 0: checked, 1: unchecked
    image = cv2.imread(src_image_file)

    # Predict on image
    results = MODEL.predict(source=image, conf=0.2, iou=0.3)
    boxes = results[0].boxes

    BOX_COLORS = {
        "unchecked": (255, 0, 0),
        "checked": (0, 128, 0),
    }

    bboxes = []
    for box in boxes:
        cls_label_numeric = box.cls[0].item()
        cls = 'unchecked' if cls_label_numeric == 1.0 else 'checked'

        start_box = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
        end_box = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
        line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
        # line_thickness = 2
        image = cv2.rectangle(
            img=image,
            pt1=start_box,
            pt2=end_box,
            color=BOX_COLORS[cls],
            thickness=line_thickness
        )

        bbox = {
            "bounding-box": {
                "x1": int(box.xyxy[0][0]),
                "y1": int(box.xyxy[0][1]),
                "x2": int(box.xyxy[0][2]),
                "y2": int(box.xyxy[0][3])
            },
            "label": cls
        }
        bboxes.append(bbox)
    cv2.imwrite(target_image_file, image)

    return image, bboxes


annotated_image, bounding_boxes = infer_from_model(
    model_dir='/app/data/ultralytics/trained-models',
    src_image_file='/app/trainer/2.png',
    target_image_file='/app/data/test.png'
)

# show image in notebook
# from matplotlib import pyplot as plt
# plt.rcParams["figure.figsize"] = (20, 30)
# plt.imshow(annotated_image)
# plt.show()
