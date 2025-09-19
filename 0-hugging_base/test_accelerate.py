import sys
import subprocess
import pkg_resources
import os

def run_cmd(cmd):
    try:
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return result.strip()
    except subprocess.CalledProcessError as e:
        return e.output.strip()

print("==== Python & Pip 路径 ====")
print("Python:", sys.executable)
print("Pip:", run_cmd("which pip"))

print("\n==== accelerate 信息 ====")
try:
    accel = pkg_resources.get_distribution("accelerate")
    print("accelerate 版本:", accel.version)
    print("安装路径:", accel.location)
except Exception as e:
    print("accelerate 未安装:", e)

print("\n==== transformers & torch 信息 ====")
for pkg in ["transformers", "torch"]:
    try:
        dist = pkg_resources.get_distribution(pkg)
        print(f"{pkg} 版本: {dist.version}, 路径: {dist.location}")
    except Exception as e:
        print(f"{pkg} 未安装: {e}")

print("\n==== 检查多个 accelerate 副本 ====")
site_packages = run_cmd("python -m site --user-site")
all_accel = run_cmd("pip show accelerate | grep Location")
print("pip show accelerate:", all_accel)

# 用 pip list 查找重复
print("\n所有 accelerate 版本:")
print(run_cmd("pip list | grep accelerate"))

print("\n==== 建议 ====")
print("1. 确认 accelerate 版本 >= 0.26.0")
print("2. 确认 transformers 版本和 torch 匹配 (建议 pip install -U transformers[torch])")
print("3. 确认 Python 和 pip 在同一个虚拟环境 (conda/venv)")
