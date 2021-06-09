from .utils import listify
from functools import partial


class ListContainer():
    def __init__(self, items):
        self.items = listify(items)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self.items[idx]
        if isinstance(idx[0], bool):  # idx为bool列表，返回idx中值为true的位置对应的元素
            assert len(idx) == len(self)  # bool mask
            return [obj for m, obj in zip(idx, self.items) if m]
        return [self.items[i] for i in idx]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __setitem__(self, i, obj):
        self.items[i] = obj

    def __delitem__(self, i):
        del(self.items[i])

    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self) > 10:
            res = res[:-1] + '...]'
        return res


class Hook():
    def __init__(self, module, fun, mode, idx):
        if mode == 'forward':
            self.hook = module.register_forward_hook(partial(fun, self))
        else:
            self.hook = module.register_backward_hook(partial(fun, self))
        m_str = str(module).replace('\n', '').replace(' ', '')
        self.name = f"{idx}-{m_str}"

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


class Hooks(ListContainer):
    def __init__(self, model, fun, mode):
        """
        Args:
            model: pytorch module
            fun: hook function
            mode: forward hook or backward hook
        """
        if mode == 'forward':
            modules = [(i, m) for i, m in enumerate(model.modules()) if len(list(m.children())) == 0]
        else:
            modules = [(i, m) for i, m in enumerate(model.modules()) if len(list(m.children())) == 0 and self.param_num(m) > 0]
        super().__init__([Hook(m, fun, mode, i) for i, m in modules])

    def param_num(self, module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self:
            h.remove()
