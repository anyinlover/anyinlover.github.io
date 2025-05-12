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


def activate_patch():
    sys.meta_path.insert(0, PatchModuleFinder(ALIAS_MAP, REPLACE_MAP))
