import cv2
import numpy as np
from paddleocr import PaddleOCR
import os
from pathlib import Path

def extract_roi_text(image_path, roi_coords):
    """
    이미지에서 ROI 영역을 추출하고 PaddleOCR을 사용하여 영어 & 숫자 텍스트를 인식합니다.
    
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
    
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=True, gpu_id=0)
    result = ocr.ocr(roi, cls=True)
    
    extracted_text = []
    if result and result[0]:
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            if confidence > 0.5:
                extracted_text.append(text)
    
    return extracted_text

def process_single_image(image_path, roi_coords):
    """
    단일 이미지에서 ROI 텍스트 추출을 수행합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        roi_coords (tuple): ROI 좌표 (x1, y1, x2, y2)
    """
    print(f"이미지 처리: {image_path}")
    print(f"ROI 좌표: x1={roi_coords[0]}, y1={roi_coords[1]}, x2={roi_coords[2]}, y2={roi_coords[3]}")
    print(f"ROI 크기: width={roi_coords[2]-roi_coords[0]}, height={roi_coords[3]-roi_coords[1]}")
    print("-" * 50)
    
    extracted_text = extract_roi_text(image_path, roi_coords)
    
    if extracted_text:
        print(f"인식된 텍스트 ({len(extracted_text)}개):")
        for i in range(0, len(extracted_text), 4):
            row_texts = extracted_text[i:i+4]
            print("  " + "  ".join(row_texts))
    else:
        print("인식된 텍스트가 없습니다.")
    
    print("-" * 30)

def main():
    roi_coords = (379, 154, 608, 943)
    image_path = "POST_EVT_4/page_1.png"
    
    print("PaddleOCR을 사용한 ROI 텍스트 인식 시작")
    print("GPU 사용: 활성화")
    print("=" * 50)
    
    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    
    process_single_image(image_path, roi_coords)
    
    print("\n이미지 처리 완료!")

if __name__ == "__main__":
    main()
