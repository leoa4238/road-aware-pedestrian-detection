# PC방(초기화되는 PC)에서 매번 실행하는 환경 설치 스크립트. 약 15분.
# 사용법: PowerShell에서  powershell -ExecutionPolicy Bypass -File setup_pcbang.ps1
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = "Tls12"
$work = "C:\work"
New-Item -ItemType Directory -Force $work | Out-Null

if (-not (Test-Path "$work\python312\python.exe")) {
    Invoke-WebRequest "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe" -OutFile "$work\python-installer.exe"
    Start-Process "$work\python-installer.exe" -ArgumentList "/quiet InstallAllUsers=0 PrependPath=0 Include_test=0 Include_launcher=0 TargetDir=$work\python312" -Wait
}
if (-not (Test-Path "$work\mingit\cmd\git.exe")) {
    Invoke-WebRequest "https://github.com/git-for-windows/git/releases/download/v2.49.0.windows.1/MinGit-2.49.0-64-bit.zip" -OutFile "$work\mingit.zip"
    Expand-Archive "$work\mingit.zip" -DestinationPath "$work\mingit" -Force
}
$env:Path = "$work\python312;$work\mingit\cmd;$env:Path"

if (-not (Test-Path "$work\rapd")) { git clone https://github.com/leoa4238/road-aware-pedestrian-detection.git "$work\rapd" }
if (-not (Test-Path "$work\venv")) { python -m venv "$work\venv" }
$py = "$work\venv\Scripts\python.exe"
& $py -m pip install --upgrade pip -q
& $py -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 -q
& $py -m pip install -r "$work\rapd\aihub\requirements.txt" -q
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
Write-Host "준비 완료. 이후 명령은 $py 로 실행하세요."
