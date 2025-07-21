import cv2
import numpy as np
from paddleocr import PaddleOCR
import os
from pathlib import Path
import csv
from datetime import datetime

def preprocess_image(image):
    """
    OCR 인식률 향상을 위한 이미지 전처리를 수행합니다.
    
    Args:
        image: 원본 이미지
    
    Returns:
        preprocessed_image: 전처리된 이미지
    """
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 노이즈 제거 (가우시안 블러)
    denoised = cv2.GaussianBlur(gray, (1, 1), 0)
    
    # 적응형 히스토그램 평활화로 대비 향상
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 이진화 (Otsu's method)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 모폴로지 연산으로 텍스트 선명화
    kernel = np.ones((1, 1), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 약간의 가우시안 블러로 텍스트 경계 부드럽게
    final = cv2.GaussianBlur(morphed, (1, 1), 0)
    
    return final

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
    
    # 전처리 수행
    preprocessed_roi = preprocess_image(roi)
    
    try:
        ocr = PaddleOCR(
            lang='en', 
            device='gpu',
            use_angle_cls=False,
            show_log=False
        )
        
        # 전처리된 이미지로 OCR 수행
        result = ocr.ocr(preprocessed_roi, cls=True)
        
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

def save_to_csv(extracted_text, page_num, csv_writer):
    """
    추출된 텍스트를 CSV 파일에 연속적으로 저장합니다.
    
    Args:
        extracted_text (list): 저장할 텍스트 리스트
        page_num (int): 페이지 번호
        csv_writer: CSV writer 객체
    """
    try:
        for i in range(0, len(extracted_text), 4):
            row_texts = extracted_text[i:i+4]
            csv_writer.writerow(["  ".join(row_texts)])
        
        print(f"페이지 {page_num}: {len(extracted_text)}개 텍스트 저장 완료")
    except Exception as e:
        print(f"CSV 파일 저장 중 오류 발생: {e}")

def process_single_image(image_path, roi_coords, page_num, csv_writer):
    """
    단일 이미지에서 ROI 텍스트 추출을 수행하고 CSV에 저장합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        roi_coords (tuple): ROI 좌표 (x1, y1, x2, y2)
        page_num (int): 페이지 번호
        csv_writer: CSV writer 객체
    """
    print(f"\n=== 페이지 {page_num} 처리 중 ===")
    
    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return False
    
    extracted_text = extract_roi_text(image_path, roi_coords)
    
    if extracted_text:
        print(f"인식된 텍스트 ({len(extracted_text)}개):")
        for i in range(0, len(extracted_text), 4):
            row_texts = extracted_text[i:i+4]
            print("  " + "  ".join(row_texts))
        
        save_to_csv(extracted_text, page_num, csv_writer)
        return True
    else:
        print("인식된 텍스트가 없습니다.")
        return False

def process_pipeline(start_page=2, end_page=51):
    """
    페이지 2번부터 51번까지 순차적으로 처리하는 파이프라인을 실행합니다.
    
    Args:
        start_page (int): 시작 페이지 번호
        end_page (int): 종료 페이지 번호
    """
    roi_coords = (346, 161, 635, 940)
    base_path = "POST_EVT_4"
    
    print(f"=== OCR 파이프라인 시작 ===")
    print(f"처리 범위: 페이지 {start_page} ~ {end_page}")
    print(f"ROI 좌표: {roi_coords}")
    print(f"기본 경로: {base_path}")
    print(f"전처리 과정: 그레이스케일 변환 → 노이즈 제거 → 대비 향상 → 이진화 → 모폴로지 연산")
    
    output_filename = "my_type1.csv"
    
    successful_pages = 0
    failed_pages = 0
    
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            for page_num in range(start_page, end_page + 1):
                image_path = f"{base_path}/page_{page_num}.png"
                
                if process_single_image(image_path, roi_coords, page_num, writer):
                    successful_pages += 1
                else:
                    failed_pages += 1
        
        print(f"\n=== 파이프라인 완료 ===")
        print(f"CSV 파일 저장 완료: {output_filename}")
        print(f"성공: {successful_pages}개 페이지")
        print(f"실패: {failed_pages}개 페이지")
        print(f"총 처리: {end_page - start_page + 1}개 페이지")
        
    except Exception as e:
        print(f"파이프라인 실행 중 오류 발생: {e}")

def main():
    process_pipeline(start_page=2, end_page=51)

if __name__ == "__main__":
    main()
