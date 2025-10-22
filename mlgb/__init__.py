# setup.py
from setuptools import setup, find_packages
setup(
    name="mlgb",              # 包名（需在 PyPI 全局唯一）
    version="0.1.0",          # 每次发版都要递增
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],
)