import yaml
import os
import os.path as osp


def load_yaml(yaml_path, encoding='utf8'):
    """load yaml file.
    yaml file content:
        data_root: &root ./data                 # anchor define
        path: !path [*root, filename]           # use alias
        str: !str [*root, filename]
        items:
        - 1
        - 2
        - 3
        key: 
        key1: value1
        key2: value2
    load result:
        {
            'data_root': './data',
            'path': './data/filename', 
            'str': './datafilename', 
            'items': [1, 2, 3], 
            'key': {
                'key1':'value1',
                'key2': 'value2'
            }
        }
    """
    def path_fn(loader, node):
        items = loader.construct_sequence(node)
        return os.path.sep.join(items)

    def str_fn(loader, node):
        items = loader.construct_sequence(node)
        return ''.join([str(item) for item in items])
    
    def echo_fn(loader, node):
        seq = loader.construct_sequence(node)
        return seq[0]

    yaml.add_constructor('!path', path_fn)
    yaml.add_constructor('!join', path_fn)  # forward compatible
    yaml.add_constructor('!str', str_fn)
    yaml.add_constructor('!cat', str_fn)    # forward compatible
    yaml.add_constructor('!echo', echo_fn)

    with open(yaml_path, 'r', encoding=encoding) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return ddict(cfg)


def check_path(path, create=True):
    """检查路径是否存在"""
    if not osp.exists(path):
        if create:
            print(f'Create path "{path}"!')
            os.mkdir(path)
        else:
            raise Exception(f'Path "{path}" does not exists!')


def check_paths(*paths, create=True):
    """检查多个路径是否存在"""
    for path in paths:
        check_path(path, create)


class ddict(dict):
    """
    可以通过“.”访问的字典。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        self[k] = ddict(v)
                    else:
                        self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    self[k] = ddict(v)
                else:
                    self[k] = v

    def __getattr__(self, key):
        try:
            value = self[key]
            return value
        except KeyError:
            raise Exception(f'KeyError! The key "{key}" does not exists!')

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self.__dict__[key]
