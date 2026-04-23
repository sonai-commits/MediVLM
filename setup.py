from setuptools import find_packages, setup

setup(
    name="medivlm",
    version="0.1.0",
    description="MediVLM: Vision Language Model for Radiology Report Generation (EMNLP Findings 2025)",
    author="Debanjan Goswami, Ronast Subedi, Shayok Chakraborty",
    license="MIT",
    packages=find_packages(exclude=("tests", "scripts", "configs", "outputs", "checkpoints")),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "torchvision>=0.16",
        "transformers>=4.40",
        "tokenizers>=0.15",
        "sentencepiece>=0.1.99",
        "numpy>=1.24",
        "pillow>=10.0",
        "pyyaml>=6.0",
        "tqdm>=4.66",
        "scikit-learn>=1.3",
        "nltk>=3.8",
        "rouge-score>=0.1.2",
    ],
    extras_require={
        "metrics": ["bert-score>=0.3.13", "sacrebleu>=2.4"],
        "dev": ["pytest>=7", "pytest-xdist"],
    },
)
