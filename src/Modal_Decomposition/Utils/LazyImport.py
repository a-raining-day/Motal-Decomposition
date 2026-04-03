"""
Realize Lazy Import
"""

class LazyImport(type):
    def __getattr__(cls, name):
        try:
            Module = __import__(name)
            setattr(cls, name, Module)  # Bind Module to cls.name

            return Module

        except ImportError:
            raise ImportError(f"Please pip install {name}")


class Lib(metaclass=LazyImport):
    """
    Access the heavy lib via this class.
    """
    pass