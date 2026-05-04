from setuptools import setup, find_packages

setup(
    name="mlx-de",
    version="0.2.0",
    description="Differentiable ODE solvers for Apple MLX, ported from torchdiffeq",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["mlx>=0.31.0"],
    extras_require={"torch": ["torch"]},
    python_requires=">=3.10",
)
