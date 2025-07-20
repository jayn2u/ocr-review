import cv2
import numpy as np
from paddleocr import PaddleOCR
import os
from pathlib import Path

def extract_roi_text(image_path, roi_coords):
    """
    이미지에서 ROI 영역을 추출하고 PaddleOCR을 사용하여 텍스트를 인식합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        roi_coords (tuple): ROI 좌표 (x1, y1, x2, y2)
    
    Returns:
        list: 인식된 텍스트 리스트
    """
    x1, y1, x2, y2 = roi_coords
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return []
    
    roi = image[y1:y2, x1:x2]
    
    try:
        ocr = PaddleOCR(
            lang='en', 
            device='gpu',
            show_log=False
        )
        
        result = ocr.ocr(roi, cls=True)
        
        extracted_text = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                confidence = line[1][1]
                if confidence > 0.6:
                    extracted_text.append(text)
        
        return extracted_text
    except Exception as e:
        print(f"OCR 처리 중 오류 발생: {e}")
        return []

def process_single_image(image_path, roi_coords):
    """
    단일 이미지에서 ROI 텍스트 추출을 수행합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        roi_coords (tuple): ROI 좌표 (x1, y1, x2, y2)
    """
    extracted_text = extract_roi_text(image_path, roi_coords)
    
    if extracted_text:
        print(f"인식된 텍스트 ({len(extracted_text)}개):")
        for i in range(0, len(extracted_text), 4):
            row_texts = extracted_text[i:i+4]
            print("  " + "  ".join(row_texts))
    else:
        print("인식된 텍스트가 없습니다.")

def main():
    roi_coords = (346, 161, 635, 940)
    image_path = "POST_EVT_4/page_2.png"
    
    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    
    process_single_image(image_path, roi_coords)

if __name__ == "__main__":
    main()
