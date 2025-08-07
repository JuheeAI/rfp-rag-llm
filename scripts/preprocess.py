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
INPUT_DIR = os.path.join(BASE_DATA_DIR, "data_pdf") # 원본 파일들이 있는 폴더 (예: /home/data/data_pdf)

# 전처리된 데이터와 로그를 저장할 최종 상위 폴더
# 이 폴더(preprocess)는 필요시 자동으로 생성
PREPROCESS_OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "preprocess")

# JSON 파일이 저장될 최종 경로: /home/data/preprocess/json
OUTPUT_DIR = os.path.join(PREPROCESS_OUTPUT_DIR, "json")
# 로그 파일이 저장될 최종 경로: /home/data/preprocess/logs
LOG_DIR = os.path.join(PREPROCESS_OUTPUT_DIR, "logs")

# 필요한 모든 디렉토리 미리 생성 (preprocess, preprocess/json, preprocess/logs 모두 생성됨)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# --- 로그 기록 함수 ---
def log_message(msg, log_file, level="INFO"):
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - [{level}] - {msg}\n")

    print(f"{timestamp} - [{level}] - {msg}")


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
        log_message(f"공고 번호 결측치 발견. Unknown_{idx}로 자동 부여.", log_file, "WARNING")
    else:
        row['공고번호_결측'] = False

    if pd.isna(row.get('공고 차수')):
        row['공고 차수'] = -1
        row['공고차수_결측'] = True
        log_message("공고 차수 결측치 발견. -1로 대체.", log_file, "WARNING")
    else:
        row['공고차수_결측'] = False

    if pd.isna(row.get('입찰 참여 시작일')):
        row['입찰 참여 시작일'] = "미정"
        row['입찰참여시작일_결측'] = True
        log_message("입찰 참여 시작일 결측치 발견. '미정'으로 대체.", log_file, "WARNING")
    else:
        row['입찰참여시작일_결측'] = False

    if pd.isna(row.get('입찰 참여 마감일')):
        row['입찰 참여 마감일'] = "미정"
        row['입찰참여마감일_결측'] = True
        log_message("입찰 참여 마감일 결측치 발견. '미정'으로 대체.", log_file, "WARNING")
    else:
        row['입찰참여마감일_결측'] = False

    # --- 사업 금액 처리 로직 개선 ---
    amount = row.get('사업 금액')
    if pd.isna(amount):
        row['사업 금액'] = 57000000
        row['사업금액_결측'] = True
        log_message("사업 금액 결측치 발견. 5,700만원으로 수기 대체.", log_file, "WARNING")
    else:
        try:
            # 콤마, 공백, 한글('원' 등)을 제거하고 숫자로 변환
            cleaned_amount = str(amount).replace(',', '').strip()

            # 한글 '억', '조' 단위 처리 추가
            if '억' in cleaned_amount:
                value, unit = cleaned_amount.split('억')
                value = float(value) * 100000000
                if unit:
                    # '억' 다음에 '만' 단위가 올 경우 처리
                    unit_value = unit.replace(',', '').replace('원', '').strip()
                    if unit_value:
                        value += float(unit_value) * 10000
                amount = value
            elif '조' in cleaned_amount:
                value, unit = cleaned_amount.split('조')
                amount = float(value) * 1000000000000 + float(unit.replace(',', '').replace('원', '').strip()) * 100000000
            else:
                amount = float(cleaned_amount)

            if amount < 1000000:
                row['사업 금액'] = "협의 예정"
                row['사업금액_결측'] = True
                log_message("사업 금액 이상치(<100만원) 발견. '협의 예정'으로 대체.", log_file, "WARNING")
            else:
                row['사업 금액'] = int(amount) # 정수형으로 저장
                row['사업금액_결측'] = False
        except Exception as e:
            row['사업 금액'] = 57000000
            row['사업금액_결측'] = True
            log_message(f"사업 금액 파싱 오류. 기본값으로 대체 (5,700만원): {e}", log_file, "WARNING")

    return row

# --- PDF에서 텍스트, 표, 이미지 메타데이터 추출  ---
def extract_from_pdf(pdf_path, log_file=None):
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    pages_data = []

    for page_num in range(num_pages):
        page = doc[page_num]
        text = page.get_text("text")

        images_on_page = page.get_images(full=False)
        image_count = len(images_on_page)
        has_images = image_count > 0

        camelot_tables = []
        try:
            camelot_result = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='lattice')
            camelot_tables = [table.df.astype(str).values.tolist() for table in camelot_result]
        except Exception as e:
            log_message(f"Camelot 표 추출 실패 (페이지 {page_num + 1}): {e}", log_file, "WARNING")

        img_tables = []
        if not camelot_tables:
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img_file:
                    page_image = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1, dpi=300)[0]
                    page_image.save(temp_img_file.name)
                    image_temp_path = temp_img_file.name

                doc_img = PDF(image_temp_path)
                extracted_img_tables = doc_img.extract_tables()

                if not isinstance(extracted_img_tables, list):
                    extracted_img_tables = []

                for table in extracted_img_tables:
                    if hasattr(table, 'content') and table.content:
                        rows = [[str(cell.text) for cell in row] for row in table.content]
                        img_tables.append(rows)

                if os.path.exists(image_temp_path):
                    os.remove(image_temp_path)

            except Exception as e:
                log_message(f"img2table 표 추출 실패 (페이지 {page_num + 1}): {e}", log_file, "WARNING")

        normalized_tables = camelot_tables if camelot_tables else img_tables

        pages_data.append({
            "page": page_num + 1,
            "text": clean_text(text),
            "tables": normalized_tables,
            "has_images": has_images,
            "image_count": image_count
        })

    return pages_data

# --- 전체 처리 함수 ---
def process_all(csv_path, input_dir, output_dir, log_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    global_log_file = os.path.join(log_dir, "global_log.txt")
    log_message(f"CSV 로딩 중: {csv_path}", global_log_file)

    df = pd.read_csv(csv_path)
    results = []

    # --- CSV 파일의 HWP 정보를 PDF로 업데이트하는 코드 추가 ---
    hwp_count = df['파일형식'].eq('hwp').sum()
    if hwp_count > 0:
        log_message(f"CSV 내 HWP 파일 {hwp_count}건을 PDF로 변경합니다.", global_log_file, "INFO")
        # '파일형식' 컬럼의 'hwp'를 'pdf'로 변경
        df['파일형식'] = df['파일형식'].replace('hwp', 'pdf')
        # '파일명' 컬럼의 확장자를 '.hwp'에서 '.pdf'로 변경
        df['파일명'] = df['파일명'].str.replace('.hwp', '.pdf', case=False, regex=False)
        df['파일명'] = df['파일명'].str.strip()  # 08.05 업로드 실패 건을 위해 추가 
        log_message("CSV 업데이트 완료.", global_log_file, "INFO")
    # --------------------------------------------------------

    for idx, row in df.iterrows():
        try:
            cleaned_row = handle_missing_values(row.copy().to_dict(), idx, global_log_file)

            # 결측치 정보와 나머지 CSV 정보를 분리
            missing_info = {
                "공고번호_결측": cleaned_row.pop('공고번호_결측', False),
                "공고차수_결측": cleaned_row.pop('공고차수_결측', False),
                "입찰참여시작일_결측": cleaned_row.pop('입찰참여시작일_결측', False),
                "입찰참여마감일_결측": cleaned_row.pop('입찰참여마감일_결측', False),
                "사업금액_결측": cleaned_row.pop('사업금액_결측', False),
            }

            # --- 여기서 '텍스트' 필드와 같은 PDF 콘텐츠 필드를 제거 ---
            pdf_related_keys = ['텍스트'] # PDF 전체 텍스트를 담는 필드명을 지정
            for key in pdf_related_keys:
                cleaned_row.pop(key, None)

            csv_metadata = cleaned_row

            file_name = unicodedata.normalize('NFD', csv_metadata.get('파일명', ''))
            file_format = csv_metadata.get('파일형식', '').lower()
            input_path = os.path.join(input_dir, file_name)

            if file_format != 'pdf':
                log_message(f"지원하지 않는 파일 형식 (hwp 등): {file_format} - {file_name}", global_log_file, "WARNING")
                continue

            if not os.path.exists(input_path):
                log_message(f"파일 없음: {input_path}", global_log_file, "ERROR")
                continue

            pages_data = extract_from_pdf(input_path, global_log_file)

            doc_result = {
                "csv_metadata": csv_metadata,
                "missing_values": missing_info,
                "pdf_data": pages_data
            }

            results.append(doc_result)

            safe_filename = sanitize_filename(f"{csv_metadata.get('공고 번호', '')}_{csv_metadata.get('사업명', '')}")
            output_json_path = os.path.join(output_dir, f"{safe_filename}.json")
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(doc_result, f, ensure_ascii=False, indent=2)

        except Exception as e:
            log_message(f"처리 중 예외 발생 - 파일명: {row.get('파일명', 'Unknown')}, 오류: {str(e)}", global_log_file, "ERROR")

    try:
        cleaned_df = pd.DataFrame(results)
        if not cleaned_df.empty:
            df_csv = cleaned_df['csv_metadata'].apply(pd.Series)
            df_missing = cleaned_df['missing_values'].apply(pd.Series)

            final_df = pd.concat([df_csv, df_missing], axis=1)

            cleaned_csv_path = os.path.join(output_dir, "data_list_cleaned.csv")
            final_df.to_csv(cleaned_csv_path, index=False)
            log_message(f"전처리 완료. 결측치 정보 포함된 CSV 저장: {cleaned_csv_path}", global_log_file)
            log_message(f"총 {len(results)}개 문서의 JSON 저장 완료: {output_dir}", global_log_file)
        else:
            log_message("처리된 문서가 없어 cleaned_csv 파일을 생성하지 않습니다.", global_log_file, "WARNING")
    except Exception as e:
        log_message(f"CSV 저장 중 예외 발생: {str(e)}", global_log_file, "ERROR")


# --- 실행 ---
if __name__ == "__main__":
    process_all(csv_path=CSV_PATH, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, log_dir=LOG_DIR)

