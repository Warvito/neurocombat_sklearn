import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='neurocombat-sklearn',
    version='0.1.3',
    author='Walter Hugo Lopez Pinaya',
    description='Harmonizing neuroimaging data across sites. Implementation of neurocombat using sklearn format',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Warvito/neurocombat-sklearn",
    project_urls={
        "Source Code": "https://github.com/Warvito/neurocombat-sklearn",
    },
    packages=setuptools.find_packages(),
    install_requires=[
        'scikit-learn',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
