def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator
