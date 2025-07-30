#!pip install pyhwp
#!sudo apt-get update
#!sudo apt-get install -y poppler-utils
#!pip install pymupdf pdf2image pytesseract camelot-py[cv] opencv-python img2table


import os
import re
import json
import pandas as pd
from pathlib import Path
import subprocess
import fitz # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
import tempfile
import camelot
from img2table.document import PDF
from datetime import datetime
import unicodedata


# --- 경로 설정 (VM 환경) ---
BASE_DATA_DIR = "/home/data" # 원본 데이터가 있는 기본 경로

# CSV 파일과 원본 파일들이 있는 폴더 경로
CSV_PATH = os.path.join(BASE_DATA_DIR, "data_list.csv")
INPUT_DIR = os.path.join(BASE_DATA_DIR, "data") # 원본 파일들이 있는 폴더 (예: /home/data/data)

# 전처리된 데이터와 로그를 저장할 최종 상위 폴더
# 이 폴더(preprocess)는 필요시 자동으로 생성됩니다.
PREPROCESS_OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "preprocess")

# JSON 파일이 저장될 최종 경로: /home/data/preprocess/json
OUTPUT_DIR = os.path.join(PREPROCESS_OUTPUT_DIR, "json")
# 로그 파일이 저장될 최종 경로: /home/data/preprocess/logs
LOG_DIR = os.path.join(PREPROCESS_OUTPUT_DIR, "logs")

# 필요한 모든 디렉토리 미리 생성 (preprocess, preprocess/json, preprocess/logs 모두 생성됨)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)



# --- 로그 기록 함수  ---
def log_message(msg, log_file):
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {msg}\n")
    print(f"{timestamp} - {msg}")

# --- 텍스트 정제 함수  ---
def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = unicodedata.normalize('NFKC', text)
    return text

# --- 파일명 안전 처리 ---
def sanitize_filename(name):
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = re.sub(r'\s+', "_", name)
    return name.strip()[:80]

# --- 결측치 처리 ---
def handle_missing_values(row, idx, log_file):
    if pd.isna(row.get('공고 번호')):
        row['공고 번호'] = f"Unknown_{idx}"
        row['공고번호_결측'] = True
        log_message(f"공고 번호 결측치 발견. Unknown_{idx}로 자동 부여.", log_file)
    else:
        row['공고번호_결측'] = False

    if pd.isna(row.get('공고 차수')):
        row['공고 차수'] = -1
        row['공고차수_결측'] = True
        log_message("공고 차수 결측치 발견. -1로 대체.", log_file)
    else:
        row['공고차수_결측'] = False

    if pd.isna(row.get('입찰 참여 시작일')):
        row['입찰 참여 시작일'] = "미정"
        row['입찰참여시작일_결측'] = True
        log_message("입찰 참여 시작일 결측치 발견. '미정'으로 대체.", log_file)
    else:
        row['입찰참여시작일_결측'] = False

    if pd.isna(row.get('입찰 참여 마감일')):
        row['입찰 참여 마감일'] = "미정"
        row['입찰참여마감일_결측'] = True
        log_message("입찰 참여 마감일 결측치 발견. '미정'으로 대체.", log_file)
    else:
        row['입찰참여마감일_결측'] = False

    amount = row.get('사업 금액')
    if pd.isna(amount):
        row['사업 금액'] = 57000000
        row['사업금액_결측'] = True
        log_message("사업 금액 결측치 발견. 5,700만원으로 수기 대체.", log_file)
    else:
        try:
            if float(amount) < 1000000:
                row['사업 금액'] = "협의 예정"
                row['사업금액_결측'] = True
                log_message("사업 금액 이상치(<100만원) 발견. '협의 예정'으로 대체.", log_file)
            else:
                row['사업금액_결측'] = False
        except Exception as e:
            row['사업 금액'] = 57000000
            row['사업금액_결측'] = True
            log_message(f"사업 금액 파싱 오류. 기본값으로 대체 (5,700만원): {e}", log_file)

    return row


# --- HWP 텍스트 추출  ---
def extract_text_from_hwp(hwp_path, log_file):
    try:
        result = subprocess.run(['hwp5txt', hwp_path], capture_output=True, text=True)
        if result.returncode != 0:
            log_message(f"[ERROR] hwp5txt 실패: {result.stderr}", log_file)
            return None
        return result.stdout
    except Exception as e:
        log_message(f"[ERROR] hwp5txt 실행 예외: {e}", log_file)
        return None

# --- PDF에서 텍스트, 표, 이미지 추출  ---
def extract_from_pdf(pdf_path, log_file=None, threshold=30):
    def is_text_poor(text):
        return len(text.strip()) < 30

    def is_image_heavy(blocks):
        # 텍스트 블록 대비 이미지 블록 비율로 판단 (단순 개수보다 비율이 더 정확)
        text_blocks_count = sum(1 for b in blocks if b['type'] == 0)
        image_blocks_count = sum(1 for b in blocks if b['type'] == 1)
        if text_blocks_count == 0: # 텍스트 블록이 아예 없으면 이미지가 많다고 간주
            return image_blocks_count > 0
        return image_blocks_count / text_blocks_count > 0.5 # 이미지 블록이 텍스트 블록의 절반 이상이면 이미지 위주로 간주

    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    pages_data = []

    log_message(f"[INFO] 파일 열기: {pdf_path}", log_file)

    # 전체 PDF를 이미지로 미리 변환할지 결정 (페이지 수가 적을 때 효율적)
    if num_pages <= threshold:
        log_message(f"[INFO] PDF 전체 이미지 변환 중 (≤ {threshold} 페이지)", log_file)
        try:
            images_all_pages = convert_from_path(pdf_path, dpi=300)
        except Exception as e:
            log_message(f"[ERROR] 전체 이미지 변환 실패: {e}", log_file)
            images_all_pages = [None] * num_pages # 실패 시 None으로 채움
    else:
        images_all_pages = [None] * num_pages # 페이지 수가 많으면 개별 페이지 필요 시 변환

    for page_num in range(num_pages):
        page = doc[page_num]
        text = page.get_text("text")
        blocks = page.get_text("dict")["blocks"] # 텍스트 및 이미지 블록 정보

        images_info = [ {"page": page_num + 1, "bbox": b['bbox']} for b in blocks if b['type'] == 1 ]

        needs_ocr = is_text_poor(text) or is_image_heavy(blocks)

        page_image = images_all_pages[page_num] # 미리 변환된 이미지를 사용하거나 None

        # 해당 페이지 이미지가 없으면 개별적으로 변환 시도
        if page_image is None:
            try:
                log_message(f"[INFO] 페이지 {page_num + 1} 이미지 변환 중...", log_file)
                page_image = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1, dpi=300)[0]
            except Exception as e:
                log_message(f"[ERROR] 페이지 {page_num + 1} 이미지 변환 실패: {e}", log_file)
                page_image = None

        ocr_text = ""
        ocr_positions = []
        image_temp_path = None # 이미지 파일 임시 경로

        if page_image:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img_file:
                image_temp_path = temp_img_file.name
                page_image.save(image_temp_path)

            if needs_ocr:
                try:
                    # Tesseract OCR 실행
                    ocr_result = pytesseract.image_to_data(page_image, output_type=pytesseract.Output.DICT, lang='kor+eng')
                    for i in range(len(ocr_result['text'])):
                        conf = ocr_result['conf'][i]
                        # OCR 신뢰도 70 이상인 텍스트만 추출
                        if str(conf).isdigit() and int(conf) > 70:
                            text_item = ocr_result['text'][i].strip()
                            if text_item: # 빈 문자열 제외
                                x, y, w, h = (ocr_result['left'][i], ocr_result['top'][i], ocr_result['width'][i], ocr_result['height'][i])
                                ocr_text += text_item + " "
                                ocr_positions.append({"text": text_item, "bbox": [x, y, x + w, y + h]})
                    ocr_text = ocr_text.strip()
                    log_message(f"[INFO] OCR 완료: 페이지 {page_num + 1}", log_file)
                    if not text and ocr_text: # 원본 텍스트가 거의 없고 OCR 결과가 있으면 OCR 텍스트를 사용
                        text = ocr_text
                except Exception as e:
                    log_message(f"[ERROR] OCR 실패: 페이지 {page_num + 1} - {e}", log_file)

        # Camelot을 이용한 테이블 추출 시도
        camelot_tables = []
        try:
            camelot_result = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='lattice')
            # Camelot의 DataFrame을 List of Lists 형태로 변환
            # 모든 셀 값을 문자열로 통일 (pandas Dataframe의 tolist()는 타입 유지)
            camelot_tables = [table.df.astype(str).values.tolist() for table in camelot_result]
            if camelot_tables:
                log_message(f"[INFO] Camelot으로 {len(camelot_tables)}개의 테이블 추출 성공 (페이지 {page_num + 1})", log_file)
        except Exception as e:
            log_message(f"[WARN] Camelot 표 추출 실패 (페이지 {page_num + 1}): {e}", log_file)

        # img2table을 이용한 테이블 추출 시도 (Camelot에서 테이블을 찾지 못했을 때만)
        img_tables = []
        if not camelot_tables and image_temp_path: # Camelot에서 추출된 테이블이 없고 임시 이미지가 있을 때만 시도
            try:
                doc_img = PDF(image_temp_path)
                # img2table은 extract_tables() 시 자동으로 List of Lists 형태를 반환 (table.content)
                extracted_img_tables = doc_img.extract_tables()

                if not isinstance(extracted_img_tables, list):
                    log_message(f"[WARN] img2table 반환값이 리스트가 아님 (페이지 {page_num + 1})", log_file)
                    extracted_img_tables = []

                for table in extracted_img_tables:
                    if hasattr(table, 'content') and table.content: # content가 비어있지 않은지 확인
                        # img2table의 content (List of List of Cells)에서 Cell 객체의 텍스트 추출
                        rows = [[str(cell.text) for cell in row] for row in table.content]
                        img_tables.append(rows)
                if img_tables:
                    log_message(f"[INFO] img2table로 {len(img_tables)}개의 테이블 추출 성공 (페이지 {page_num + 1})", log_file)
            except Exception as e:
                log_message(f"[WARN] img2table 표 추출 실패 (페이지 {page_num + 1}): {e}", log_file)

        # 임시 이미지 파일 정리
        if image_temp_path and os.path.exists(image_temp_path):
            os.remove(image_temp_path)

        # ----------------------------------------------------------------------
        # 정규화된 테이블 리스트 생성: Camelot 우선, 없으면 img2table 사용
        normalized_tables = []
        if camelot_tables:
            normalized_tables.extend(camelot_tables)
        elif img_tables:
            normalized_tables.extend(img_tables)
        # ----------------------------------------------------------------------

        pages_data.append({
            "page": page_num + 1,
            "text": clean_text(text),
            "ocr_text": clean_text(ocr_text),
            "ocr_positions": ocr_positions,
            "images": images_info,
            "tables": normalized_tables # 정규화된 테이블 데이터
        })

    return pages_data


# --- HWP 파일 처리 ---
def process_hwp_file(hwp_path, log_file):
    text = extract_text_from_hwp(hwp_path, log_file)
    if text is None:
        text = ""
    text = clean_text(text)
    pages_data = [{
        "page": 1,
        "text": text,
        "ocr_text": "",
        "ocr_positions": [],
        "images": [],
        "tables": []
    }]
    return pages_data

# --- 전체 처리 함수 ---
def process_all(csv_path, input_dir, output_dir, log_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    global_log_file = os.path.join(log_dir, "global_log.txt")
    log_message(f"[INFO] CSV 로딩 중: {csv_path}", global_log_file)

    df = pd.read_csv(csv_path)
    results = []

    for idx, row in df.iterrows():
        log_file = os.path.join(log_dir, f"{idx:04d}_log.txt")
        log_message(f"\n=== {idx}번째 데이터 처리 시작 ===", log_file)

        try:
            log_message(f"[INFO] 결측치 처리 시작: {row['파일명']}", log_file)
            row = handle_missing_values(row, idx, log_file)
            log_message(f"[INFO] 결측치 처리 완료", log_file)

            file_name = row['파일명']
            file_format = row['파일형식'].lower()
            input_path = os.path.join(input_dir, file_name)

            log_message(f"[INFO] 입력 파일 확인: {input_path}", log_file)
            if not os.path.exists(input_path):
                log_message(f"[ERROR] 파일 없음: {input_path}", log_file)
                continue

            if file_format == 'hwp':
                log_message(f"[INFO] HWP 처리 시작", log_file)
                pages_data = process_hwp_file(input_path, log_file)
                log_message(f"[SUCCESS] HWP 파일 처리 완료: {input_path}", log_file)
            elif file_format == 'pdf':
                log_message(f"[INFO] PDF 처리 시작", log_file)
                pages_data = extract_from_pdf(input_path, log_file)
                log_message(f"[SUCCESS] PDF 파일 처리 완료: {input_path}", log_file)
            else:
                log_message(f"[WARNING] 지원하지 않는 파일 형식: {file_format}", log_file)
                continue

            doc_result = {
                "공고번호": row['공고 번호'],
                "사업명": row.get('사업명', ''),
                "파일명": file_name,
                "페이지별_데이터": pages_data,
                "결측치정보": {
                    "공고번호_결측": row['공고번호_결측'],
                    "공고차수_결측": row['공고차수_결측'],
                    "입찰참여시작일_결측": row['입찰참여시작일_결측'],
                    "입찰참여마감일_결측": row['입찰참여마감일_결측'],
                    "사업금액_결측": row['사업금액_결측'],
                }
            }
            results.append(doc_result)

            safe_filename = sanitize_filename(f"{row['공고 번호']}_{row.get('사업명', '')}")
            output_json_path = os.path.join(output_dir, f"{safe_filename}.json")
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(doc_result, f, ensure_ascii=False, indent=2)
            log_message(f"[SUCCESS] JSON 파일 저장 완료: {output_json_path}", log_file)

        except Exception as e:
            log_message(f"[ERROR] 처리 중 예외 발생 - {file_name}: {str(e)}", log_file)

    try:
        cleaned_csv_path = os.path.join(output_dir, "data_list_cleaned.csv")
        df.to_csv(cleaned_csv_path, index=False)
        print(f"[DONE] 전처리 완료. 결측치 포함 CSV 저장: {cleaned_csv_path}")
        print(f"[DONE] 총 {len(results)}개 문서의 JSON 저장 완료: {output_dir}")
    except Exception as e:
        print(f"[ERROR] CSV 저장 중 예외 발생: {str(e)}")


# --- 실행 ---
if __name__ == "__main__":
    process_all(csv_path=CSV_PATH, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, log_dir=LOG_DIR)

