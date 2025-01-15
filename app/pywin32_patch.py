import platform
from pathlib import Path
import sys
import pkgutil
import importlib
import os

def _patch_win32_deps(max_attempts=5):
    if not platform.system() == "Windows" or os.environ.get('BAZEL_PYWIN_REMAP') is None:
        return

    import ctypes
    def find_pywin32_dir(path: Path) -> Path:
        for path in path.iterdir():
            if "pywin32" in path.name and path.is_dir():
                return path
        return None

    pywin_lib_path = find_pywin32_dir(Path.cwd().parent)
    pywin32_system32_path = f"{pywin_lib_path}\\site-packages\\pywin32_system32"
    os.add_dll_directory(f"{pywin_lib_path}\\site-packages\\win32")
    os.add_dll_directory(pywin32_system32_path)
    os.add_dll_directory(f"{pywin_lib_path}\\site-packages\\pythonwin")
    os.add_dll_directory(f"{pywin_lib_path}\\site-packages\\isapi")

    # Load and initialize pywintypes
    def initialize_pywintypes():
        # Find the pywintypes DLL in the pywin32_system32 directory
        pywintypes_dll = None
        for file in Path(pywin32_system32_path).iterdir():
            if file.name.startswith("pywintypes") and file.suffix == ".dll":
                pywintypes_dll = file
                break
        
        if pywintypes_dll:
            # Load the DLL using ctypes
            dll = ctypes.CDLL(str(pywintypes_dll))
            print(str(dll))
            # Add a reference in sys.modules for pywintypes
            sys.modules['pywintypes'] = dll
            import win32._win32sysloader
            sys.modules['_win32sysloader'] = win32._win32sysloader
            import win32.lib.pywintypes
            sys.modules['pywintypes'] = win32.lib.pywintypes
            print(f"Loaded pywintypes from {pywintypes_dll}")
        else:
            print("Failed to locate pywintypes DLL")

    # Initialize pywintypes before remapping modules
    initialize_pywintypes()

    # Discover and remap all submodules in the `win32` namespace
    def remap_win32_modules():
        namespaces_to_remap = ["win32", "win32.lib"]
        deferred_remap = []  # Modules to remap later
        attempt = 0

        for namespace in namespaces_to_remap:
            for finder, name, ispkg in pkgutil.iter_modules(importlib.import_module(namespace).__path__, namespace + "."):
                alias = name[len(namespace) + 1:]
                if alias in sys.modules:
                    continue  # Skip already remapped modules
                if name == "win32.lib.win32traceutil":
                    continue # skip this as it remaps output

                try:
                    module = importlib.import_module(name)
                    sys.modules[alias] = module
                    print(f"Remapped {name} to alias {alias}")
                except ImportError as e:
                    # Handle failures and defer loading
                    print(f"Failed to import {name}: {e}")
                    deferred_remap.append((name, alias))

        while attempt < max_attempts:
            # If no new deferred items are added, we're done
            if len(deferred_remap) == 0:
                print("Everything mapped correctly")
                break
            else:
                print(f"List of remappings: {deferred_remap}")
                new_deferred = []
                for alias, name in deferred_remap:
                    print(f"Retrying mapping of {name}")
                    try:
                        module = importlib.import_module(name)
                        sys.modules[alias] = module
                        print(f"Remapped {name} to alias {alias}")
                    except ImportError as e:
                        # Handle failures and defer loading
                        print(f"Failed to import {name}: {e}")
                        new_deferred.append((name, alias))

            # Retry with newly deferred items
            deferred_remap = new_deferred
            attempt += 1

        # Log unresolved modules if any remain after all attempts
        if deferred_remap:
            print(f"Unresolved modules after {max_attempts} attempts:")
            for name, alias in deferred_remap:
                print(f"  {name} (alias: {alias})")

    remap_win32_modules()

_patch_win32_deps()