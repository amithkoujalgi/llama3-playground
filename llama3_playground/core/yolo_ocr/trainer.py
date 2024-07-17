# Prepare Dataset here: https://universe.roboflow.com/search?q=checkbox
import os
import shutil

import ruamel
from ultralytics import YOLO

os.environ['WANDB_MODE'] = 'disabled'

dataset_cfg_yaml = '/app/models/ultralytics/yolo-v8/training-dataset-config.yaml'
model_yaml = "/app/models/ultralytics/yolo-v8/yolov8s.yaml"
base_model_file = "/app/models/ultralytics/yolo-v8/yolov8s.pt"
target_model_dir = "/app/data/ultralytics/trained-models"

config, ind, bsi = ruamel.yaml.util.load_yaml_guess_indent(open(dataset_cfg_yaml))

yaml = ruamel.yaml.YAML()
yaml.indent(mapping=ind, sequence=ind, offset=bsi)

with open(dataset_cfg_yaml, 'w') as fp:
    yaml.dump(config, fp)

# Due to lack of computational resources, we will use YOLOv8s model with to train the initial checkpoint
model = YOLO(model_yaml).load(base_model_file)

# Train with minimum 200 epochs
trainer_result = model.train(
    data=dataset_cfg_yaml,
    epochs=200,
    patience=75,
    imgsz=640,
    workers=8,
    batch=-1,
    name="checkbox_detection_train1",
    mixup=0.3,
    copy_paste=0.3,
    device=0,
    save_dir=target_model_dir
)

# move trained mode from './runs/detect/checkbox_detection_train1' to target dir
trained_model_intermediate_path = str(trainer_result.save_dir)
shutil.move(src=trained_model_intermediate_path, dst=target_model_dir)
