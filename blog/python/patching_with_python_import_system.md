---
title: Patching with Python's Import System
date: 2025-05-13 13:07:00
tags:
  - python
---

In many scenarios, we need to apply patches to third-party libraries in Python. A common approach is to use "monkey patching." However, monkey patching is not a perfect solution because it dynamically changes attributes after a module has been imported. Sometimes, the module being modified might have already been imported before the changes take effect, causing the monkey patch to not work as expected.

We need to find a way to modify modules as early as possible. A better method is to leverage Python's import system to achieve this. For detailed documentation on Python's import system, please refer to the [official documentation](https://docs.python.org/3/reference/import.html). In short, Python imports a module in three steps:

1.  Search for the module using a Finder.
2.  Create the module using a Loader.
3.  Bind the module in the current namespace.

In step 1, we can hook into `sys.meta_path` to create a custom finder, which can return a different module specification (module spec) based on a given module name. In step 2, we can create a new loader for a specific module, which replaces certain attributes (functions, classes, variables) of the module before the created module is returned.

Therefore, with this approach, we can replace an entire module or its attributes when the module is first imported. Since `sys.modules` acts as a cache, each module is created only once. Consequently, after a module is modified, it will never change again, which is exactly what we expect.

Below is an example code:

```python
import importlib
import importlib.abc
import importlib.util
import sys

ALIAS_MAP = {"d": "patch.d"}
REPLACE_MAP = {"pkg.a.a_1": {"a_1_func1": ("patch.pkg.a.a_1", "a_1_func1_wrapper")}}


class AttrReplacingLoader(importlib.abc.Loader):
    def __init__(self, original_loader, replace_map):
        self.original_loader = original_loader
        self.replace_map = replace_map

    def create_module(self, spec):
        if hasattr(self.original_loader, "create_module"):
            return self.original_loader.create_module(spec)
        return None

    def exec_module(self, module):
        self.original_loader.exec_module(module)
        to_replace = self.replace_map[module.__spec__.name]
        for attr_name, (new_module_name, new_attr_name) in to_replace.items():
            new_module = importlib.import_module(new_module_name)
            if hasattr(module, attr_name):
                setattr(module, attr_name, getattr(new_module, new_attr_name)(getattr(module, attr_name)))


class PatchModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self, alias_map, replace_map):
        self.alias_map = alias_map
        self.replace_map = replace_map

    def find_spec(self, fullname, path, target=None):
        if fullname in self.alias_map:
            original_name = self.alias_map[fullname]
            original_spec = importlib.util.find_spec(original_name)
            if original_spec:
                return original_spec

        if fullname in self.replace_map:
            if self in sys.meta_path:
                sys.meta_path.remove(self)
            original_spec = None
            try:
                original_spec = importlib.util.find_spec(fullname, path)
            finally:
                if self not in sys.meta_path:
                    sys.meta_path.insert(0, self)

            if original_spec and original_spec.loader:
                return importlib.util.spec_from_loader(
                    fullname,
                    AttrReplacingLoader(original_spec.loader, self.replace_map),
                    origin=original_spec.origin,
                    is_package=original_spec.submodule_search_locations is not None,
                )
        return None

sys.meta_path.insert(0, PatchModuleFinder(ALIAS_MAP, REPLACE_MAP))
```