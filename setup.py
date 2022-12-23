from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_args = dict(
    name="deep_tabular_augmentation", 
    version="0.5.3",
    author="Lasse Schmidt",
    author_email="lasse.schmidt@live.de",
    description="Small package for data augmentation via Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lschmiddey/deep_tabular_augmentation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

if __name__ == '__main__':
    setup(**setup_args)