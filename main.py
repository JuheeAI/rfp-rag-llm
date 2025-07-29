# main.py
 
import multiprocessing
import uvicorn
import subprocess
import sys
import os


##### 백엔드 서버 실행 함수 #####
# 9000번 포트로 실행
def run_fastapi():
    from src.backend import app
    print(">>>>> Start Backend Server <<<<<")
    uvicorn.run(app, host="0.0.0.0", port=9000)

##### 프론트엔드 서버 실행 함수 #####
# 8501번 포트로 실행
def run_streamlit():
    print(">>>>> Start Frontend Server <<<<<")
    frontend_path = os.path.join("src", "frontend.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", frontend_path, "--server.port", "8501"])


##### 터미널 명령어 실행 #####
if __name__ == "__main__":
    # 프로세스 생성
    fastapi_process = multiprocessing.Process(target=run_fastapi)
    streamlit_process = multiprocessing.Process(target=run_streamlit)

    # 프로세스 시작
    fastapi_process.start()
    streamlit_process.start()

    # 끝날 때까지 대기
    fastapi_process.join()
    streamlit_process.join()
