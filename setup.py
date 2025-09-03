from setuptools import setup, find_packages

setup(
    name="glucoguard",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scikit-learn==1.3.2",
        "xgboost==2.0.3",
        "optuna==3.5.0",
    ],
    author="Your Name",
    description="AI-powered glucose prediction with Empatica E4",
    python_requires=">=3.8",
)
