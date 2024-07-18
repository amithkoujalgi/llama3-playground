import argparse
import json
import os.path
import shutil
import traceback
import uuid

import cv2
import easyocr
import numpy as np
from PIL import Image
from fillpdf import fillpdfs
from prettytable import PrettyTable
from ultralytics import YOLO

from llama3_playground.core.config import Config

model_dir = "/app/data/ultralytics/trained-models"


class YOLOv8OCR:
    def __init__(self, run_dir: str, pdf_file_path: str, model_path: str):
        self._run_dir = run_dir
        self._pdf_file_path = pdf_file_path
        self._model_path = model_path

    def _pdf_to_images(self) -> []:
        """
        Uses `pymupdf` lib to convert PDF to images.
        """
        import pymupdf
        from pymupdf import Matrix

        unprocessed_images_dir = os.path.join(self._run_dir, 'unprocessed-images')
        os.makedirs(unprocessed_images_dir, exist_ok=True)

        image_files = []
        with pymupdf.open(self._pdf_file_path) as doc:
            for page_num in range(len(doc)):
                page_img_file = f'{unprocessed_images_dir}/page-{str(page_num)}.png'
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=Matrix(3.0, 3.0))
                pix.save(page_img_file)
                image_files.append(page_img_file)
        return image_files

    def process(self):
        processed_images_dir = os.path.join(self._run_dir, 'processed-images')
        os.makedirs(processed_images_dir, exist_ok=True)

        raw_image_files = self._pdf_to_images()

        all_bboxes = []

        page_num = 0

        processed_images = []
        for raw_image_file in raw_image_files:
            processed_image_file = f'{processed_images_dir}/{os.path.basename(raw_image_file)}'
            _cv2_image, bboxes = self._infer_from_model(
                src_image_file=raw_image_file,
                target_image_file=processed_image_file
            )
            all_bboxes.append({
                page_num: bboxes
            })
            page_num += 1
            processed_images.append(processed_image_file)

        output_file = os.path.join(self._run_dir, 'text-result.txt')
        self._extract_and_align_text(processed_images, output_file)

        ocr_data = {
            "ocr-data": {},
            "checkboxes": all_bboxes
        }
        with open(f'{self._run_dir}/ocr-result.json', 'w') as f:
            f.write(json.dumps(ocr_data, indent=4))

    def _infer_from_model(self, src_image_file: str, target_image_file: str):
        # 0: checked, 1: unchecked
        _model_instance = YOLO(f"{self._model_path}/weights/best.pt")
        image = cv2.imread(src_image_file)

        results = _model_instance.predict(source=image, conf=0.2, iou=0.3)
        boxes = results[0].boxes

        box_colors = {
            "unchecked": (255, 0, 0),
            "checked": (0, 128, 0),
        }

        bboxes = []
        for box in boxes:
            cls_label_numeric = box.cls[0].item()

            cls = 'unchecked' if cls_label_numeric == 1.0 else 'checked'
            replacement_text = '[NO]' if cls == 'unchecked' else '[YES]'
            bool_flag = False if cls == 'unchecked' else True

            start_box = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
            end_box = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
            line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
            line_thickness = 10

            option_rect_fill_color = (255, 255, 255)
            option_color = (0, 0, 0)
            option_font_scale = 0.5
            option_thickness = 2
            option_font = cv2.FONT_ITALIC
            option_rect_diff = 3
            option_rect_thickness = -1

            x = start_box[0]
            y = start_box[1]
            w = end_box[0] - x
            h = end_box[1] - y

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # make checkbox background white
            # image = cv2.rectangle(
            #     img=image,
            #     pt1=start_box,
            #     pt2=end_box,
            #     color=option_rect_fill_color,
            #     # color=box_colors[cls],
            #     thickness=line_thickness
            # )

            # image = cv2.rectangle(
            #     img=image,
            #     pt1=start_box,
            #     pt2=end_box,
            #     color=(255, 255, 255),
            #     thickness=line_thickness
            # )

            cv2.rectangle(image, (x - option_rect_diff, y - option_rect_diff),
                          (x + w + option_rect_diff, y + h + option_rect_diff), option_rect_fill_color,
                          option_rect_thickness)

            # place text YES/NO over the checkbox
            image = cv2.putText(image, replacement_text, (x, y + h - option_rect_diff), option_font, option_font_scale,
                                option_color,
                                option_thickness)
            bbox = {
                "bounding-box": {
                    "x1": int(box.xyxy[0][0]),
                    "y1": int(box.xyxy[0][1]),
                    "x2": int(box.xyxy[0][2]),
                    "y2": int(box.xyxy[0][3])
                },
                "label": cls,
                "checked": bool_flag
            }
            bboxes.append(bbox)
        cv2.imwrite(target_image_file, image)

        return image, bboxes

    def _extract_and_align_text(self, images, output_file, line_threshold=20, paragraph_threshold=50):
        reader = easyocr.Reader(['en'])
        aligned_text = ""
        paragraph_gaps = []

        for img in images:
            img_ = Image.open(img)
            img_ = np.array(img_)  # Convert PIL Image to numpy array for EasyOCR
            # input_to_ocr_file_name = f"{self._intermediate_results_path}/input_to_ocr.jpg"
            # print(f"Writing to: {input_to_ocr_file_name}")
            # cv2.imwrite(input_to_ocr_file_name, img)
            results = reader.readtext(img_, detail=1)

            # Group results by lines based on vertical position
            lines = []
            for bbox, text, _ in results:
                top_left = bbox[0]
                found = False
                for line in lines:
                    if abs(line['top'] - top_left[1]) < line_threshold:  # Line threshold
                        line['content'].append((bbox, text))
                        found = True
                        break
                if not found:
                    lines.append({'top': top_left[1], 'content': [(bbox, text)]})

            # Sort each line by horizontal position and detect paragraph breaks
            prev_bottom = 0
            for line in sorted(lines, key=lambda x: x['top']):
                line['content'].sort(key=lambda x: x[0][0][0])  # Sort by X coordinate
                text_line = ' '.join([text for _, text in line['content']])

                # Check for paragraph breaks
                paragraph_gap = line['top'] - prev_bottom
                paragraph_gaps.append(paragraph_gap)
                if paragraph_gap > np.percentile(paragraph_gaps, 90):  # Use a higher percentile for paragraph breaks
                    aligned_text += "\n\n---SEPARATOR---\n\n"
                aligned_text += text_line + "\n"

                prev_bottom = line['content'][-1][0][2][1]  # Update bottom position from the last element of the line

        with open(output_file, 'w') as f:
            f.write(aligned_text)
        print(f"Writing to: {output_file}")


def print_cli_args(cli_args: argparse.Namespace):
    print("Using the following config:")
    t = PrettyTable(['Config Key', 'Specified Value'])
    t.align["Config Key"] = "r"
    t.align["Specified Value"] = "l"
    for k, v in cli_args.__dict__.items():
        t.add_row([k, v])
    print(t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A utility to run OCR and extract text from a PDF file.')
    parser.add_argument(
        '-r',
        '--run-id',
        type=str,
        dest='run_id',
        help='A run ID (string) to create a folder with the OCR run data. Example: "123"',
        required=False,
        default=str(uuid.uuid4())
    )
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument(
        '-f',
        '--pdf-path',
        type=str,
        dest='pdf_file_path',
        help='Path to the source PDF file. Example: "/tmp/input.pdf"',
        required=True
    )
    args: argparse.Namespace = parser.parse_args()
    print_cli_args(cli_args=args)

    run_id = args.run_id
    pdfFile = args.pdf_file_path

    outDir = f'{Config.ocr_runs_dir}/{run_id}'
    if not os.path.isdir(outDir):
        print(f"Creating out directory: {outDir}")
        os.makedirs(outDir)

    shutil.copyfile(pdfFile, f'{outDir}/{os.path.basename(pdfFile)}')

    fillpdfs.flatten_pdf(pdfFile, pdfFile, as_images=False)

    try:
        ocr = YOLOv8OCR(run_dir=outDir, pdf_file_path=pdfFile, model_path=model_dir)
        ocr.process()
        with open(os.path.join(outDir, 'RUN-STATUS'), 'w') as f:
            f.write("success")
    except Exception as e:
        error_str = traceback.format_exc()
        print(f"OCR Error: {e}. Cause: {error_str}")
        with open(os.path.join(outDir, 'error.log'), 'w') as f:
            f.write(error_str)
        with open(os.path.join(outDir, 'RUN-STATUS'), 'w') as f:
            f.write("failure")
