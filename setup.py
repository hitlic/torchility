from torchility import __vsersion__
import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchility",
    version=__vsersion__,
    author="hitlic",
    author_email="liuchen.lic@gmail.com",
    license='MIT',
    description="",
    long_description=long_description,
    url="",
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='',
    packages=setuptools.find_packages(
        include=['torchility'],
        exclude=['.git',
                 '.git/*',
                 '.gitignore',
                 '.vscode',
                 '.vscode/*',
                 'torchility/__pycache__/*',
                 'test.py',
                 'test'
                 'test/*',
                 'lightning_logs'
                 ]),
    py_modules=[],  # any single-file Python modules that aren’t part of a package
    install_requires=['torch > 1.7', 'pytorch-lightning>1.1'],
    python_requires='>=3.6'
)
