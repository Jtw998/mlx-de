from setuptools import setup, find_packages

setup(
    name="mlx-de",
    version="0.1.0",
    description="ODE solvers for MLX — port of torchdiffeq",
    packages=find_packages(),
    install_requires=["mlx>=0.31.0"],
    python_requires=">=3.10",
)
