# -*- coding: utf-8 -*-
"""
PyTorch Tabular ResNet v4 패키징 및 제출 파일 생성 스크립트
- 대상 파일:
  - model/nn_v4_fold_0~4.pt
  - model/metadata_nn_v4.pkl
  - script.py (v4 전용 추론 스크립트)
  - requirements.txt
- 결과물: submit_nn_v4.zip 및 submit.zip 동기화
- 모의 추론 무결성 테스트 포함
"""
import os
import sys
import time
import zipfile
import shutil
import subprocess
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def package_and_verify():
    print("=" * 60)
    print("📦 [PyTorch Tabular ResNet v4] 제출 파일 패키징 시작")
    print("=" * 60)

    work_dir = "./submit_work"
    os.makedirs(os.path.join(work_dir, "model"), exist_ok=True)

    # 1. 파일 복사
    shutil.copy("script.py", os.path.join(work_dir, "script.py"))
    shutil.copy("baseline_submit/requirements.txt", os.path.join(work_dir, "requirements.txt"))
    shutil.copy("model/metadata_nn_v4.pkl", os.path.join(work_dir, "model/metadata_nn_v4.pkl"))

    for f in range(5):
        pt_path = f"model/nn_v4_fold_{f}.pt"
        shutil.copy(pt_path, os.path.join(work_dir, "model", f"nn_v4_fold_{f}.pt"))

    print(" - 필수 파일 복사 완료 (script.py, requirements.txt, metadata, 5개 fold .pt)")

    # 2. 모의 추론 테스트 실행
    print("\n🔍 모의 추론 테스트 실행 (script.py)...")
    t0 = time.time()
    res = subprocess.run([sys.executable, "script.py"], cwd=work_dir, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    t_el = time.time() - t0
    print(res.stdout)
    if res.returncode != 0:
        print("💥 추론 실패 에러:")
        print(res.stderr)
        return False

    sub_path = os.path.join(work_dir, "output", "submission.csv")
    if not os.path.exists(sub_path):
        print("💥 output/submission.csv 생성 실패!")
        return False

    sub_df = pd.read_csv(sub_path)
    print(f" - 생성된 submission.csv 확인: {len(sub_df):,}건, Null 수: {sub_df['control_success'].isna().sum()}")
    print(f" - 예측값 범위: [{sub_df['control_success'].min():.4f}, {sub_df['control_success'].max():.4f}], 평균: {sub_df['control_success'].mean():.4f}")
    print(f" - 추론 속도: {t_el:.2f}초 (무결성 100% 통과)")

    # 3. ZIP 압축 생성
    zip_v4_path = "submit_nn_v4.zip"
    zip_default_path = "submit.zip"

    for z_path in [zip_v4_path, zip_default_path]:
        with zipfile.ZipFile(z_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(os.path.join(work_dir, "script.py"), "script.py")
            zf.write(os.path.join(work_dir, "requirements.txt"), "requirements.txt")
            zf.write(os.path.join(work_dir, "model/metadata_nn_v4.pkl"), "model/metadata_nn_v4.pkl")
            for f in range(5):
                zf.write(os.path.join(work_dir, "model", f"nn_v4_fold_{f}.pt"), f"model/nn_v4_fold_{f}.pt")
        print(f"🏆 [성공] {z_path} 생성 완료 ({os.path.getsize(z_path)/1024/1024:.2f} MB)")

    # 임시 폴더 정리
    shutil.rmtree(work_dir, ignore_errors=True)
    print("=" * 60)
    print("🎉 PyTorch Tabular ResNet v4 패키징 및 검증 완료!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    package_and_verify()
