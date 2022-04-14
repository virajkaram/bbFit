import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bbFit",
    version="0.0.1",
    author="Viraj Karambelkar",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="astronomy blackbody MCMC",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires='>=3.8',
    install_requires=[
        "astropy",
        "numpy",
        "emcee",
        "matplotlib"
    ]
)
