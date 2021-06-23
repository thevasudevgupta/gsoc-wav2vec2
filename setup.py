__version__ = "0.0.1"

import setuptools


with open("readme.md", "r", encoding="utf-8") as file:
    long_description = file.read()

with open("requirements.txt", "r", encoding="utf-8") as file:
    install_requires = [p.strip() for p in file]

setuptools.setup(
    name="wav2vec2",
    version=__version__,
    author="Vasudev Gupta",
    author_email="7vasudevgupta@gmail.com",
    description="TensorFlow implementation of Wav2Vec2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache",
    url="https://github.com/VasudevGupta7/gsoc-wav2vec2",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
)
