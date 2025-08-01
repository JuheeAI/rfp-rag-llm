import pyautogui
import pyperclip
import time
import os

hwp_dir = "./hwp_data"
pdf_dir = "./data_pdf"

files = [f for f in os.listdir(hwp_dir) if f.lower().endswith('.hwp')]
print("HWP files:", files)

for f in files:
    filepath = os.path.join(hwp_dir, f)
    print(f"Opening: {filepath}")
    
    os.startfile(filepath)
    time.sleep(20)

    pyautogui.hotkey('ctrl', 'p')
    time.sleep(5)

    pyautogui.press('enter')
    time.sleep(4)

    filename_only = os.path.splitext(f)[0] + '.pdf'
    print(f"저장할 파일 이름: {filename_only}")
    pyperclip.copy(filename_only)
    print("클립보드 내용:", pyperclip.paste())
    pyautogui.hotkey('ctrl', 'v')            
    time.sleep(0.5)
    pyautogui.press('enter')
    time.sleep(8)
