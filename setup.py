from setuptools import setup, find_packages

setup(
    name="sentiment_bert_pipeline",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "torch",
        "transformers",
        "datasets"
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8", "isort"]
    },
)
