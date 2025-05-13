from setuptools import setup, find_packages

setup(
    name='sppml',
    version='0.1.0',
    description='Student Performance Prediction ML Package',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'joblib'
    ],
    include_package_data=True,
    package_data={
        'sppml': ['models/*.pkl']
    },
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
)
