import inspect
from functools import wraps

__all__ = ['memoize_components']


def memoize_components(static_method=True):
    """ A decorator factory for memoizing the components of an influence matrix or other method with components

    Parameters
    ----------
    static_method: bool, optional (True)
        True if the object to be decorated is an instance or class method

    Notes
    -----
    This returns a decorator that can be used to memoize a callable which finds components. The callable MUST:

    Have it's first argument be the component
    components must be hashable

    The cache is a dict with the same keys as previously passed components, when any of the other input arguments change
    the cache is deleted

    The wrapped callable will have the additional attributes:

    cache : dict
        All of the cached values, use cache.clear() to remove manually
    spec : list
        The other arguments passed to the callable (if any of these change the cache is cleared)
    """
    if not isinstance(static_method, bool):
        raise ValueError('memoize_components is a decorator factory, it cannot be applied as a decorator directly.'
                         ' static_method argument must be a bool')

    def outer(fn):
        # non local variables spec is a list to ensure it's mutable

        sig = inspect.signature(fn)

        if static_method:
            spec = []
            cache = []

            @wraps(fn)
            def inner(component, *args, **kwargs):
                nonlocal cache, spec, sig
                new_spec = sig.bind(None, *args, **kwargs)
                new_spec.apply_defaults()
                try:
                    index = spec.index(new_spec)
                except ValueError:
                    spec.append(new_spec)
                    cache.append(dict())
                    index = len(cache) - 1
                if component not in cache[index]:
                    cache[index][component] = fn(component, *args, **kwargs)
                return cache[index][component]
        else:
            spec = dict()
            cache = dict()

            @wraps(fn)
            def inner(self, components, *args, **kwargs):
                nonlocal cache, spec, sig
                if self.name not in cache:
                    cache[self.name] = []
                    spec[self.name] = []
                new_spec = sig.bind(None, None, *args, **kwargs)
                new_spec.apply_defaults()
                try:
                    index = spec[self.name].index(new_spec)
                except ValueError:
                    spec[self.name].append(new_spec)
                    cache[self.name].append(dict())
                    index = len(cache[self.name]) - 1
                comps_to_find = [c for c in components if c not in cache[self.name][index]]
                if comps_to_find:
                    cache[self.name][index].update(fn(self, comps_to_find, *args, **kwargs))
                return {comp: cache[self.name][index][comp] for comp in components}

        inner.cache = cache
        inner.spec = spec

        return inner

    return outer
