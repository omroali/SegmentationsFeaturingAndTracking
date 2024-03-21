from setuptools import find_packages
from setuptools import setup


setup(
    name="seg-assignment",
    version="1.0.0",
    description="package for code development related to the Computer Vision module",
    author="Omar Ali",
    author_email="28587497@students.lincoln.ac.uk",
    url="https://github.com/olseda20",
    install_requires=[
        line.strip() for line in open('requirements.txt')
    ],
    packages=find_packages(exclude=("test*")),
    entry_points={
        "console_scripts": [
            "hello-world = SegmentationFeatureAndTracking.main:main",
        ],
    },
)
