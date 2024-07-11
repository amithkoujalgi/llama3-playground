# !pip install easyocr boto3 langchain langchain_community langchain_core fastembed chromadb pdf2image tiktoken
import argparse
import json
import os
import shutil
import sys
import traceback
from config import Config

import cv2
import easyocr
import numpy as np
from prettytable import PrettyTable
from fillpdf import fillpdfs


def pdf_to_images(pdf_path: str, pdf_pages_to_images_dir: str) -> []:
    """
    Uses `pdf2image` lib to convert PDF to images.
    Note: It is observed that some images from the filled PDFs misses some filled form fields.
    """
    import pdf2image
    images = pdf2image.convert_from_path(pdf_path)
    # images = pdf2image.convert_from_bytes(open(pdf_path, 'rb').read())
    image_files = []
    for page_num in range(len(images)):
        page_img_file = f'{pdf_pages_to_images_dir}/page-{str(page_num)}.jpg'
        images[page_num].save(page_img_file, 'JPEG')
        image_files.append(page_img_file)
    return image_files


def pdf_to_images2(pdf_path: str, pdf_pages_to_images_dir: str) -> []:
    """
    Uses `pymupdf` lib to convert PDF to images.
    """
    import pymupdf
    from pymupdf import Matrix

    image_files = []
    with pymupdf.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page_img_file = f'{pdf_pages_to_images_dir}/page-{str(page_num)}.png'
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=Matrix(3.0, 3.0))
            pix.save(page_img_file)
            image_files.append(page_img_file)
    return image_files


def update_checkboxes_in_image(image_path):
    def _check_proximity(prev_points, curr_point) -> bool:
        """
        Returns true if the points are not close to each other, false otherwise.
        """
        x_distance_threshold = 50  # adjust threshold as needed
        y_distance_threshold = 50  # adjust threshold as needed

        last_point = prev_points[-1:]
        if len(last_point) == 0:
            return True
        last_point = last_point[0]
        x_diff = abs(float(curr_point['x']) - float(last_point['x']))
        y_diff = abs(float(curr_point['y']) - float(last_point['y']))
        if x_diff > x_distance_threshold or y_diff > y_distance_threshold:
            return True
        return False

    def _check_if_point_square(item):
        """
        Returns true if the detected point is square-like polygon, false otherwise.
        """
        tolerance = 0.2  # adjust tolerance as needed
        _x = item['x']
        _y = item['y']
        _w = item['w']
        _h = item['h']
        # Check if the aspect ratio is close to 1 (i.e., w ≈ h)
        aspect_ratio = _w / _h
        if 1 - tolerance <= aspect_ratio <= 1 + tolerance:
            return True
        return False

    # Reading input image
    image = cv2.imread(image_path)

    # Converting BGR to Grayscale
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize image
    th1, img_bin = cv2.threshold(gray_scale, 180, 255, cv2.THRESH_OTSU)

    # Defining kernels
    lWidth = 1
    lineMinWidth = 25

    kernal1 = np.ones((lWidth, lWidth), np.uint8)
    kernal1h = np.ones((1, lWidth), np.uint8)
    kernal1v = np.ones((lWidth, 1), np.uint8)

    kernal6 = np.ones((lineMinWidth, lineMinWidth), np.uint8)
    kernal6h = np.ones((1, lineMinWidth), np.uint8)
    kernal6v = np.ones((lineMinWidth, 1), np.uint8)

    # Finding horizontal lines
    img_bin_h = cv2.morphologyEx(~img_bin, cv2.MORPH_CLOSE, kernal1h)
    img_bin_h = cv2.morphologyEx(img_bin_h, cv2.MORPH_OPEN, kernal6h)

    # Finding vertical lines
    img_bin_v = cv2.morphologyEx(~img_bin, cv2.MORPH_CLOSE, kernal1v)
    img_bin_v = cv2.morphologyEx(img_bin_v, cv2.MORPH_OPEN, kernal6v)

    # Function to fix binarized images
    def fix(img):
        img[img > 127] = 255
        img[img < 127] = 0
        return img

    # Final binarized image
    img_bin_final = fix(fix(img_bin_h) | fix(img_bin_v))

    # Getting connected components and their statistics
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)

    boxes_identified = []
    filled_checkboxes = []
    unfilled_checkboxes = []

    # Iterating over each bounding box and checking if it's filled
    for x, y, w, h, area in stats[2:]:
        # Extract the region of interest (ROI) from the binary image
        roi = img_bin[y:y + h, x:x + w]

        # Calculate number of filled pixels (white pixels)
        num_white_pixels = np.sum(roi == 255)

        # Calculate total number of pixels in the ROI
        total_pixels = roi.shape[0] * roi.shape[1]

        # Calculate fill percentage
        fill_percentage = (num_white_pixels / total_pixels) * 100

        # Determine if the rectangle is filled based on some threshold
        is_unfilled = fill_percentage > 80  # Adjust the threshold as per your requirement

        point = {
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
        }

        # if _check_proximity(prev_points=boxes_identified, curr_point=point) and _check_if_point_square(point):
        if _check_if_point_square(point):
            option_color = (0, 0, 0)
            option_font_scale = 0.5
            option_thickness = 2
            option_font = cv2.FONT_ITALIC
            option_yes = '[YES]'
            option_no = '[NO]'

            option_rect_fill_color = (255, 255, 255)
            option_rect_thickness = -1

            option_rect_diff = 3
            # Draw the rectangle on the original image based on the fill status
            if is_unfilled:
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(image, (x - option_rect_diff, y - option_rect_diff),
                              (x + w + option_rect_diff, y + h + option_rect_diff), option_rect_fill_color,
                              option_rect_thickness)
                cv2.putText(image, option_no, (x - w, y + h), option_font, option_font_scale, option_color,
                            option_thickness)
                unfilled_checkboxes.append(point)
            else:
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(image, (x - option_rect_diff, y - option_rect_diff),
                              (x + w + option_rect_diff, y + h + option_rect_diff), option_rect_fill_color,
                              option_rect_thickness)
                cv2.putText(image, option_yes, (x - w, y + h), option_font, option_font_scale, option_color,
                            option_thickness)
                filled_checkboxes.append(point)

        boxes_identified.append(point)

    # Save the marked image
    cv2.imwrite(image_path, image)
    return filled_checkboxes, unfilled_checkboxes


def print_dict_as_table(dict_to_print):
    t = PrettyTable(['Field', 'Value'])
    t.align["Field"] = "l"
    t.align["Value"] = "l"
    for k in dict_to_print:
        t.add_row([k, dict_to_print[k]])
    print(t)


def run_ocr(image_file: str):
    def _convert_to_xywh(coords):
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]

        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)

        x = x_min
        y = y_min
        w = x_max - x_min
        h = y_max - y_min

        return {
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
        }

    _data = []
    reader = easyocr.Reader(['en'], model_storage_directory=os.environ['EASY_OCR_MODELS_DIR'], download_enabled=False)
    result = reader.readtext(image_file)
    for item in result:
        coordinates, text, score = item
        _data.append({
            "text": text,
            "coordinates": _convert_to_xywh(coordinates),
            "confidence-score": score
        })
    return _data


def pdf_to_text(pdf_path: str, runs_dir: str):
    """Run OCR to extract text from PDF file."""
    pdf_pages_to_images_dir = f'{runs_dir}/pdf-images'
    pdf_pages_to_text_dir = f'{runs_dir}/pdf-ocr'

    pages_data = []
    if not os.path.isdir(pdf_pages_to_images_dir):
        print("Creating images directory...")
        os.makedirs(pdf_pages_to_images_dir)
    if not os.path.isdir(pdf_pages_to_text_dir):
        print("Creating OCR directory...")
        os.makedirs(pdf_pages_to_text_dir)

    images = pdf_to_images2(pdf_path=pdf_path, pdf_pages_to_images_dir=pdf_pages_to_images_dir)

    for i in range(len(images)):
        print(f"Updating checkboxes in page {i}...")
        update_checkboxes_in_image(image_path=images[i])
        print(f"Running OCR on page {i}...")
        page_txt = run_ocr(image_file=images[i])
        with open(os.path.join(pdf_pages_to_text_dir, f'ocr-page-result-{i}.json'), 'w') as f:
            f.write(json.dumps(page_txt, default=str, indent=4))
        pages_data.append({
            "page": i,
            "ocr-data": page_txt
        })
    print(f'OCR for {len(images)} pages done.')
    pages_ocr_data_file = os.path.join(runs_dir, f'ocr-result.json')
    ocr_json_data = json.dumps(pages_data, default=str, indent=4)
    with open(pages_ocr_data_file, 'w') as f:
        f.write(ocr_json_data)
    print(f'Wrote OCR JSON result to: {pages_ocr_data_file}')
    pages_text = ''
    for page_data in pages_data:
        page_num = page_data['page']
        page_ocr_data = page_data['ocr-data']
        page_full_text = ''
        for section in page_ocr_data:
            page_text = section['text']
            page_full_text = f'{page_full_text}\n\n{page_text}'
        pages_text = f'{pages_text}\n\n---PAGE {page_num}---\n\n{page_full_text}'

    pages_text_data_file = os.path.join(runs_dir, 'text-result.txt')
    with open(pages_text_data_file, 'w') as f:
        f.write(pages_text)
    print(f'Wrote OCR text result to: {pages_text_data_file}')
    return pages_text, pages_text_data_file, ocr_json_data, pages_ocr_data_file


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
        required=True
    )
    parser.add_argument(
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
        pdf_pages_text_data, pages_text_data_file_path, ocr_data, ocr_data_file_path = pdf_to_text(
            pdf_path=pdfFile,
            runs_dir=outDir
        )
        with open(os.path.join(outDir, 'RUN-STATUS'), 'w') as f:
            f.write("success")
    except Exception as e:
        error_str = traceback.format_exc()
        print(f"OCR Error: {e}. Cause: {error_str}")
        with open(os.path.join(outDir, 'error.log'), 'w') as f:
            f.write(error_str)
        with open(os.path.join(outDir, 'RUN-STATUS'), 'w') as f:
            f.write("failure")
