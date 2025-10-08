from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mbti-lstm",
    version="0.1.0",
    description="Классификация типов личности MBTI на основе LSTM по текстам социальных сетей",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mueqee/MBTI-LSTM",
    project_urls={
        "Отслеживание ошибок": "https://github.com/mueqee/MBTI-LSTM/issues",
        "Документация": "https://github.com/mueqee/MBTI-LSTM/tree/main/docs",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mbti-train=scripts.train:main",
            # "mbti-evaluate=scripts.evaluate:main",  # Ещё не реализовано
            # "mbti-predict=scripts.predict:main",  # Ещё не реализовано
        ],
    },
)

