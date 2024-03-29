from torchility import __version__
import setuptools

with open("README.md", 'r', encoding="utf8") as fh:
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
    # install_requires=['torch>=2.0', 'pytorch-lightning==2.0', 'torchmetrics==0.11.4', 'matplotlib>=3.3','pyyaml>=5.4', 'tensorboard>2.9'],
    install_requires=['torch>=2.0', 'pytorch-lightning==2.0', 'torchmetrics==0.11.4', 'matplotlib>=3.3','pyyaml>=5.4', 'tensorboardX==2.6'],
    python_requires='>=3.8'
)
