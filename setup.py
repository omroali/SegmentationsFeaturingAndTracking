from setuptools import find_packages
from setuptools import setup


setup(
    name="SegmentationsFeaturingAndTracking",
    version="1.0.0",
    description="package for code development related to the Computer Vision module",
    author="Omar Ali",
    author_email="28587497@students.lincoln.ac.uk",
    url="https://github.com/olseda20",
    install_requires=[line.strip() for line in open("requirements.txt")],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "trying = segmentation.main:main",
        ],
    },
)
