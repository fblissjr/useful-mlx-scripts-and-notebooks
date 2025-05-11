import argparse
import importlib
import inspect
import os
import sys
import traceback
import pkgutil

def print_env_info(project_root_to_add=None):
    """
    Prints Python environment information.
    If project_root_to_add is provided and is a valid directory,
    it's prepended to sys.path to allow importing from that root.
    """
    print("--- Python Environment Information ---")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version.splitlines()[0]}") # First line for brevity
    print(f"Current Working Directory: {os.getcwd()}")

    if project_root_to_add:
        abs_project_root = os.path.abspath(project_root_to_add)
        if os.path.isdir(abs_project_root):
            if abs_project_root not in sys.path:
                print(f"Adding provided project root to sys.path: {abs_project_root}")
                sys.path.insert(0, abs_project_root) # Add to front for priority
            else:
                print(f"Provided project root ({abs_project_root}) is already in sys.path.")
        else:
            print(f"WARNING: Provided project root '{project_root_to_add}' ('{abs_project_root}') does not exist or is not a directory.")

    print("\nCurrent sys.path:")
    for p in sys.path:
        print(f"  - {p}")
    print("--- End Environment Information ---\n")

def resolve_target(target_path_str):
    """
    Resolves a target string (e.g., 'os.path.join', 'collections.Counter',
    'module:attribute.sub_attr') into an actual Python object.

    It first attempts to import the longest possible module name from the
    start of the target string. Then, it traverses the remaining parts
    as attributes. Handles module:attribute syntax common in some frameworks.
    """
    print(f"Attempting to resolve target: '{target_path_str}'")
    module_spec_str = target_path_str
    attrs_after_colon = None

    # Handle 'module:attribute' syntax
    if ':' in target_path_str:
        module_spec_str, attrs_after_colon = target_path_str.split(':', 1)

    parts = module_spec_str.split('.')
    module_obj = None
    attrs_to_get_from_module = []

    # Iterate backwards to find the longest valid module path
    # e.g., for "a.b.c.d", try "a.b.c", then "a.b", then "a"
    for i in range(len(parts), 0, -1):
        potential_module_name = ".".join(parts[:i])
        try:
            module_obj = importlib.import_module(potential_module_name)
            attrs_to_get_from_module = parts[i:] # Remaining parts are attributes
            break
        except ImportError:
            if i == 1: # Even the first part failed to import
                raise ImportError(f"Could not import base module '{parts[0]}' from '{target_path_str}'.")
            # Continue to try a shorter module path

    if module_obj is None: # Should be caught by the loop's ImportError
        raise ImportError(f"Failed to import any part of '{module_spec_str}' as a module.")

    current_obj = module_obj
    resolved_path_parts = [module_obj.__name__] # Start with the actual imported module name

    # Resolve attributes within the imported module (before any ':')
    for attr_name in attrs_to_get_from_module:
        try:
            current_obj = getattr(current_obj, attr_name)
            resolved_path_parts.append(attr_name)
        except AttributeError as e:
            raise AttributeError(f"Object '{'.'.join(resolved_path_parts)}' has no attribute '{attr_name}'.") from e

    # Resolve attributes specified after ':' if any (e.g., my_module:app.config)
    if attrs_after_colon:
        # For display purposes, keep the colon in the conceptual path
        display_path_after_colon = ":" + attrs_after_colon
        for attr_name in attrs_after_colon.split('.'):
            try:
                current_obj = getattr(current_obj, attr_name)
            except AttributeError as e:
                # Construct the path that failed for a clearer error
                base_resolved_path = '.'.join(resolved_path_parts)
                raise AttributeError(f"Failed to get attribute '{attr_name}' from '{base_resolved_path}:{attrs_after_colon.split('.')[0]}...'.") from e
        resolved_path_parts.append(display_path_after_colon)

    final_resolved_path = '.'.join(resolved_path_parts)
    print(f"Successfully resolved '{target_path_str}' to type {type(current_obj).__name__} (as '{final_resolved_path}').")
    return current_obj

def diagnose_import_issues(library_name_to_diagnose):
    """
    Attempts to import the specified library and its submodules.
    Reports successes, failures, and tracebacks for import errors.
    Uses pkgutil.walk_packages to discover submodules if the target is a package.
    """
    print(f"\n--- Diagnosing Import Issues for '{library_name_to_diagnose}' ---")

    print(f"\nAttempting top-level import: 'import {library_name_to_diagnose}'")
    try:
        module = importlib.import_module(library_name_to_diagnose)
        print(f"  SUCCESS: `import {library_name_to_diagnose}` successful.")
        print(f"    Location: {getattr(module, '__file__', 'Not a file module')}")
        if hasattr(module, '__path__'): # Check if it's a package
            print(f"    Package Path: {getattr(module, '__path__', 'N/A')}")

        # If it's a package, try to import its submodules
        if hasattr(module, '__path__'):
            print(f"\n  --- Attempting to import submodules of '{library_name_to_diagnose}' ---")
            found_submodules = False
            # module.__path__ might be None for some namespace packages or if misconfigured
            module_paths_to_search = getattr(module, '__path__', None)
            if module_paths_to_search is None:
                 print(f"    WARNING: Module '{library_name_to_diagnose}' has no valid __path__. Cannot discover submodules via pkgutil.")
            else:
                for _, modname, _ in pkgutil.walk_packages(module_paths_to_search, module.__name__ + '.'):
                    found_submodules = True
                    try:
                        sub_module = importlib.import_module(modname)
                        print(f"      SUCCESS: Imported {modname} (from {getattr(sub_module, '__file__', 'Unknown')})")
                    except ImportError as e_sub:
                        print(f"      FAILED: Could not import {modname}: {e_sub}")
                    except Exception as e_sub_other: # Catch other errors during submodule import
                        print(f"      FAILED: Unexpected error importing {modname}: {e_sub_other}")
                        traceback.print_exc(limit=2) # Show a bit more for unexpected errors
            if not found_submodules and module_paths_to_search:
                print(f"    No discoverable submodules found via pkgutil in {module_paths_to_search}.")

    except ImportError as e:
        print(f"  FAILED: `import {library_name_to_diagnose}` failed: {e}")
        traceback.print_exc()
        print("\n  Common issues causing import failures:")
        print("  1. The library might not be installed (e.g., `pip install library_name`).")
        print("  2. If it's a local project, its root directory might not be in `sys.path`.")
        print("     Use the `-p` or `--project-root` option to add it.")
        print("  3. Typos in the library name or module path.")
        print("  4. Dependencies of the library might be missing or broken.")
    except Exception as e:
        print(f"  FAILED: `import {library_name_to_diagnose}` encountered an unexpected error: {e}")
        traceback.print_exc()
    print(f"\n--- Finished diagnosing '{library_name_to_diagnose}' ---")

def list_apis(obj, obj_name_str, show_all=False):
    """
    Lists APIs (attributes, methods, classes, etc.) for a given Python object.
    Filters out private members (starting with '_') unless show_all is True.
    Displays type, name, a signature snippet (for callables), and first line of docstring.
    """
    print(f"\n--- APIs for '{obj_name_str}' (type: {type(obj).__name__}) ---")

    members = []
    try:
        members = inspect.getmembers(obj)
    except Exception as e: # Some objects might resist getmembers
        print(f"  Could not retrieve members for '{obj_name_str}': {e}")
        print(f"--- End of APIs for '{obj_name_str}' ---")
        return

    output_lines = []
    for name, member_obj in members:
        # Filter private members unless --all is specified
        # Dunder methods (e.g. __init__) are typically shown.
        if not show_all and name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            continue

        try:
            obj_type_str = type(member_obj).__name__
            doc_summary = ""
            # Try to get the first line of the docstring for routines and classes
            if inspect.isroutine(member_obj) or inspect.isclass(member_obj):
                doc = inspect.getdoc(member_obj)
                if doc:
                    doc_summary = doc.strip().split('\n')[0]

            # Provide a clear type indicator for common types
            type_indicator = ""
            if inspect.ismodule(member_obj): type_indicator = "[Mod]"
            elif inspect.isclass(member_obj): type_indicator = "[Cls]"
            elif inspect.isfunction(member_obj): type_indicator = "[Func]"
            elif inspect.isbuiltin(member_obj): type_indicator = "[Builtin]" # Covers funcs/methods
            elif inspect.ismethod(member_obj): type_indicator = "[Meth]"
            elif inspect.ismethoddescriptor(member_obj): type_indicator = "[MethDesc]"
            else: type_indicator = f"[{obj_type_str}]" # Fallback to raw type name

            signature_str = ""
            # Attempt to get a signature for callables (excluding classes/modules themselves)
            if callable(member_obj) and not (inspect.isclass(member_obj) or inspect.ismodule(member_obj)):
                try:
                    sig = inspect.signature(member_obj)
                    signature_str = str(sig)
                    if len(signature_str) > 60: # Truncate long signatures for display
                        signature_str = signature_str[:57] + "..."
                except (ValueError, TypeError): # Signatures aren't available for all callables (e.g., some built-ins)
                    signature_str = "(...)"

            line = f"  {type_indicator:<15} {name:<30}"
            if signature_str:
                line += f" {signature_str:<45}"
            if doc_summary:
                 line += f" # {doc_summary}"
            output_lines.append(line)
        except Exception: # Catch errors inspecting a specific member
            output_lines.append(f"  [ERROR]         {name:<30} (Error inspecting this member)")

    if not output_lines:
        print("  No APIs found or all were filtered out based on current settings.")
    else:
        # Sort alphabetically by name for consistent and readable output
        for line in sorted(output_lines, key=lambda x: x.strip().split()[1].lower()):
            print(line)
    print(f"--- End of APIs for '{obj_name_str}' ---")

def show_signature_for_obj(obj, obj_name_str):
    """
    Shows detailed signature, docstring, and other relevant info for a Python object.
    Handles modules, classes (showing __init__ signature), functions/methods, and data.
    """
    print(f"\n--- Signature/Help for '{obj_name_str}' (type: {type(obj).__name__}) ---")

    doc = inspect.getdoc(obj) # inspect.getdoc is good at finding the right docstring
    # Prefer obj.__name__ if available, otherwise use the user-provided string
    name_of_obj = getattr(obj, '__name__', obj_name_str)

    if inspect.isclass(obj):
        print(f"Class: {name_of_obj}")
        if doc:
            print("\nDocstring:\n----------\n" + doc + "\n----------")
        else:
            print("\n(No class docstring found)")

        # Show __init__ signature, as it's key to using a class
        try:
            init_method = obj.__init__
            # Avoid showing the generic object.__init__ signature unless it's 'object' itself
            # or if the class explicitly defines __init__ (even if it's `pass`)
            if obj is object or init_method is not object.__init__ or hasattr(obj, '__init__'):
                 init_sig = inspect.signature(init_method)
                 print(f"\nConstructor `__init__` signature: {init_sig}")
            else: # Class inherits __init__ from object without overriding
                 print("\nConstructor `__init__` signature: (default, inherited from `object`)")
        except (AttributeError, ValueError, TypeError): # If __init__ is not standard or no signature
            print(f"\nConstructor `__init__` signature: (Could not determine or not applicable)")

    elif inspect.isroutine(obj): # Covers functions, methods, built-in functions/methods
        print(f"Callable: {name_of_obj}")
        try:
            sig = inspect.signature(obj)
            print(f"\nSignature: {sig}")
        except (ValueError, TypeError): # Signature not available for some C-level callables
            print("\nSignature: (Not available for this type of callable, e.g., some built-ins)")

        if doc:
            print("\nDocstring:\n----------\n" + doc + "\n----------")
        else:
            print(f"\n(No Python docstring found for {name_of_obj})")

    elif inspect.ismodule(obj):
        print(f"Module: {name_of_obj}")
        if hasattr(obj, '__file__'):
            print(f"  File: {obj.__file__}")
        if hasattr(obj, '__path__'): # Indicates a package
            print(f"  Package Path(s): {obj.__path__}")

        if doc:
            print("\nDocstring:\n----------\n" + doc + "\n----------")
        else:
            print("\n(No module-level docstring found)")
        print("\nNote: For exploring module contents, consider using the 'list_apis' action.")

    else: # Handles data attributes, properties, descriptors, etc.
        print(f"Object: {obj_name_str} (Python type: {type(obj).__name__})")
        try:
            val_repr = repr(obj)
            if len(val_repr) > 120: val_repr = val_repr[:117] + "..." # Truncate very long reprs
            print(f"  Value (repr): {val_repr}")
        except Exception as e:
            print(f"  Value (repr): (Error getting repr: {e})")

        if doc: # Some non-callable objects, like properties, can have docstrings
            print("\nDocstring:\n----------\n" + doc + "\n----------")

    print(f"--- End of Signature/Help for '{obj_name_str}' ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python Library Inspection and Diagnostic Tool.",
        # RawTextHelpFormatter preserves newline characters in help messages
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "target",
        help="The target Python object to inspect.\n"
             "Examples:\n"
             "  'os'                  (module)\n"
             "  'os.path.join'        (function)\n"
             "  'collections.Counter' (class)\n"
             "  'your_module:app'     (object 'app' in 'your_module')\n"
             "  'package.module.Class.method' (method)"
    )
    parser.add_argument(
        "--action", "-a",
        choices=["diagnose", "list_apis", "show_signature"],
        help="Action to perform:\n"
             "  diagnose:         Check importability of the base library and its submodules.\n"
             "                    (Uses the base module of the target string).\n"
             "  list_apis:        List public APIs (attributes, methods, etc.) of the resolved target.\n"
             "  show_signature:   Show signature, docstring, and other details for the resolved target.\n"
             "(If not specified, action is inferred based on target resolution success and type.)"
    )
    parser.add_argument(
        "--project-root", "-p",
        help="Path to a project root directory. This path will be added to the\n"
             "beginning of sys.path to aid in importing local project modules."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include private members (starting with '_') when using 'list_apis'.\n"
             "Dunder methods (__method__) are generally shown by default."
    )

    args = parser.parse_args()

    # Display environment info first, applying --project-root if given
    print_env_info(args.project_root)

    action_to_perform = args.action
    target_str = args.target
    resolved_obj = None
    resolution_error = None

    # Attempt to resolve the target string to an object, unless the action is 'diagnose'
    # 'diagnose' works directly with the library name string.
    if action_to_perform != 'diagnose':
        try:
            resolved_obj = resolve_target(target_str)
        except Exception as e:
            resolution_error = e
            print(f"Error resolving target '{target_str}': {type(e).__name__}: {e}")
            # If resolution fails and no specific action was given, default to 'diagnose'.
            if not action_to_perform:
                print(f"Target resolution failed. Defaulting to 'diagnose' for the base module.")
                action_to_perform = 'diagnose'

    # If no action was explicitly specified by the user, infer the best one.
    if not args.action and not action_to_perform: # action_to_perform might be set above if resolution failed
        if resolved_obj:
            if inspect.ismodule(resolved_obj):
                action_to_perform = "list_apis"
            elif callable(resolved_obj) or inspect.isclass(resolved_obj):
                action_to_perform = "show_signature"
            else: # For other types like data attributes, show_signature can provide repr and doc.
                action_to_perform = "show_signature"
            print(f"\nNo action specified, inferred action: '{action_to_perform}' for target '{target_str}'.")
        else:
            # If resolution failed and we are here, it means args.action was None.
            # resolution_error should exist. Default to diagnose.
            if not resolution_error: # Should ideally not happen if resolved_obj is None
                 print(f"Warning: Target '{target_str}' not resolved, but no specific resolution error captured before action inference.")
            action_to_perform = 'diagnose'
            print(f"\nNo action specified and target '{target_str}' not resolved (or resolution failed prior), defaulting to 'diagnose'.")

    # --- Execute the determined action ---
    exit_code = 0
    if action_to_perform == "diagnose":
        # For diagnose, we operate on the potential base library name from the target string.
        # e.g., "numpy.linalg.solve" -> "numpy"; "mylib:app" -> "mylib"
        library_to_diagnose = target_str.split('.')[0].split(':')[0]
        diagnose_import_issues(library_to_diagnose)
        # Diagnose itself prints success/failure, so usually exits 0 unless internal error.
        # If the initial target resolution failed which led here, resolution_error will be set.
        if resolution_error: exit_code = 1


    elif action_to_perform == "list_apis":
        if resolved_obj:
            list_apis(resolved_obj, target_str, args.all)
        else:
            print(f"\nCannot list APIs: Target '{target_str}' was not resolved or resolution failed.")
            if resolution_error:
                 print(f"  Reason: {type(resolution_error).__name__}: {resolution_error}")
            else: # Should not happen if logic is correct
                 print(f"  Reason: Unknown, target object is None without a recorded resolution error.")
            # As a fallback, offer to diagnose the base module.
            base_module_name = target_str.split('.')[0].split(':')[0]
            print(f"\nAttempting to diagnose base library '{base_module_name}' as a fallback...")
            diagnose_import_issues(base_module_name)
            exit_code = 1 # Indicate failure to perform the requested action

    elif action_to_perform == "show_signature":
        if resolved_obj:
            show_signature_for_obj(resolved_obj, target_str)
        else:
            print(f"\nCannot show signature: Target '{target_str}' was not resolved or resolution failed.")
            if resolution_error:
                 print(f"  Reason: {type(resolution_error).__name__}: {resolution_error}")
            else:
                 print(f"  Reason: Unknown, target object is None without a recorded resolution error.")
            base_module_name = target_str.split('.')[0].split(':')[0]
            print(f"\nAttempting to diagnose base library '{base_module_name}' as a fallback...")
            diagnose_import_issues(base_module_name)
            exit_code = 1 # Indicate failure to perform the requested action

    # If there was a resolution error AND the chosen action was not 'diagnose' (which handles its own errors),
    # then the script effectively failed its primary requested task.
    if resolution_error and action_to_perform != 'diagnose':
        exit_code = 1

    sys.exit(exit_code)