import os
import shutil

import cv2
import easyocr
import numpy as np
from pdf2image import convert_from_path


class OCRBySantosh:
    def __init__(self, results_dir: str, empty_results_dir_before_run: bool = False):
        self._intermediate_results_path = results_dir
        self._empty_results_dir_before_run = empty_results_dir_before_run

    def _convert_pdf_to_images(self, pdf_file_path):
        return convert_from_path(pdf_file_path)

    def _detect_checkboxes_with_hierarchy(self, images, filled=True):

        annotated_images = []
        out_images_dir = f"{self._intermediate_results_path}/annotated-pages"
        checkbox_out_images_dir = f"{self._intermediate_results_path}/pages-with-checkboxes"
        os.makedirs(out_images_dir, exist_ok=True)
        os.makedirs(checkbox_out_images_dir, exist_ok=True)

        for index, pil_image in enumerate(images):
            checkboxes = []
            image_np = np.array(pil_image)
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # Apply morphological operations to improve contour detection
            kernel = np.ones((3, 3), np.uint8)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Find contours with hierarchy
            contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            new_image_np = image_np.copy()
            cv2.drawContours(new_image_np, contours, -1, (0, 255, 0), 2)
            new_image_file_name = f"{out_images_dir}/annotated-page-{index}.jpg"
            print(f"Writing to: {new_image_file_name}")
            cv2.imwrite(new_image_file_name, new_image_np)

            for i, contour in enumerate(contours):
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)

                    # Adjust size and aspect ratio constraints
                    if 10 < w < 40 and 10 < h < 40 and 0.8 < aspect_ratio < 1.2:
                        checkbox_roi = morph[y:y + h, x:x + w]

                        # Calculate the filled ratio
                        filled_ratio = cv2.countNonZero(checkbox_roi) / float(w * h)

                        # Determine if we are looking for filled or unfilled checkboxes
                        if (filled and filled_ratio > 0.5) or (not filled and filled_ratio < 0.5):
                            checkboxes.append({"x": x, "y": y, "w": w, "h": h})
                            # cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle

            # Save the annotated image for debugging
            if filled:
                self._replace_checkbox_with_text(image_np, checkboxes, text="YES")
            else:
                self._replace_checkbox_with_text(image_np, checkboxes, text="NO")
            annotated_images.append(image_np)
            checkbox_image_file_name = f"{checkbox_out_images_dir}/checkboxed-page-{index}.jpg"
            print(f"Writing to: {checkbox_image_file_name}")
            cv2.imwrite(f'{checkbox_image_file_name}', image_np)
        return annotated_images

    def _replace_checkbox_with_text(self, image, checkbox_coords, text="SELECTED_ONE"):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2
        color = (0, 0, 0)  # Black color for text

        for checkbox in checkbox_coords:
            x, y, w, h = checkbox['x'], checkbox['y'], checkbox['w'], checkbox['h']

            # Draw a white-filled rectangle to "erase" the checkbox
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)

            # Get the size of the text
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Overlay text on the image
            cv2.putText(image, text, (x + (w - text_width) // 2, y + h - (h - text_height) // 2),
                        font, font_scale, color, font_thickness, lineType=cv2.LINE_AA)

        return image

    def _extract_and_align_text(self, images, output_file, line_threshold=20, paragraph_threshold=50):
        reader = easyocr.Reader(['en'])
        aligned_text = ""
        paragraph_gaps = []

        for img in images:
            img = np.array(img)  # Convert PIL Image to numpy array for EasyOCR
            input_to_ocr_file_name = f"{self._intermediate_results_path}/input_to_ocr.jpg"
            print(f"Writing to: {input_to_ocr_file_name}")
            cv2.imwrite(input_to_ocr_file_name, img)
            results = reader.readtext(img, detail=1)

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
                    aligned_text += "\n\n\n"
                aligned_text += text_line + "\n"

                prev_bottom = line['content'][-1][0][2][1]  # Update bottom position from the last element of the line

        with open(output_file, 'w') as f:
            f.write(aligned_text)
        print(f"Writing to: {output_file}")

    def process(self, pdf_path):
        if self._empty_results_dir_before_run:
            shutil.rmtree(self._intermediate_results_path, ignore_errors=True)
        print(f"Emptying dir: {self._intermediate_results_path}")
        images = self._convert_pdf_to_images(pdf_path)

        # Detect and replace filled checkboxes with "YES"
        filled_annotated_images = self._detect_checkboxes_with_hierarchy(images, filled=True)
        # Detect and replace unfilled checkboxes with "NO"
        fully_annotated_images = self._detect_checkboxes_with_hierarchy(filled_annotated_images, filled=False)

        # Extract and align text using OCR
        output_file = f"{self._intermediate_results_path}/result.txt"
        self._extract_and_align_text(fully_annotated_images, output_file)


OCRBySantosh(results_dir='/Users/amithkoujalgi/Downloads/llm-extraction/new',
             empty_results_dir_before_run=True).process(
    "/Users/amithkoujalgi/Downloads/llm-extraction/XPAA Demo3 IK.pdf")
