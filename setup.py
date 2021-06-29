from torchility import __version__
import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchility",
    version=__version__,
    author="hitlic",
    author_email="liuchen.lic@gmail.com",
    license='MIT',
    description="",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/hitlic/torchility",
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='',
    packages=setuptools.find_packages(exclude=['__pycache__', '__pycache__/*']),
    py_modules=[],  # any single-file Python modules that aren’t part of a package
    install_requires=['torch>1.7', 'pytorch-lightning>1.3', 'torchmetrics>0.3', 'matplotlib>=3.3'],
    python_requires='>=3.6'
)
