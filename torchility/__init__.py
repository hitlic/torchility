from .trainer import Trainer
from .callbacks import *
name = 'torchility'
__version__ = '0.8.2'

def update(to=None, newer_than=None):
    from packaging.version import parse
    import os

    cmd = 'pip install -U torchility'
    if to is not None:
        if parse(__version__) != parse(to):
            cmd = f'pip install -U torchility=={to}'
    elif newer_than is not None:
        if parse(__version__) >= parse(newer_than):
            cmd = None
    if cmd is not None:
        os.system(cmd)
