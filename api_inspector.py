import argparse
import importlib
import inspect
import os
import sys
import traceback
import pkgutil

def print_env_info(project_root_to_add=None):
    """Prints Python environment information and optionally modifies sys.path."""
    print("--- Python Environment Information ---")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version.splitlines()[0]}")
    print(f"Current Working Directory: {os.getcwd()}")

    if project_root_to_add:
        abs_project_root = os.path.abspath(project_root_to_add)
        if os.path.isdir(abs_project_root):
            if abs_project_root not in sys.path:
                print(f"Adding provided project root to sys.path: {abs_project_root}")
                sys.path.insert(0, abs_project_root)
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
    'module:attribute.sub_attr') to a Python object.
    """
    print(f"Attempting to resolve target: '{target_path_str}'")
    module_spec_str = target_path_str
    attrs_after_colon = None

    if ':' in target_path_str:
        module_spec_str, attrs_after_colon = target_path_str.split(':', 1)

    parts = module_spec_str.split('.')
    module_obj = None
    attrs_to_get_from_module = []

    # Find the longest part of module_spec_str that is an importable module
    for i in range(len(parts), 0, -1):
        potential_module_name = ".".join(parts[:i])
        try:
            module_obj = importlib.import_module(potential_module_name)
            # print(f"  Successfully imported base module '{potential_module_name}'.")
            attrs_to_get_from_module = parts[i:]
            break
        except ImportError:
            if i == 1: # Even the first part is not a module
                # print(f"  Failed to import '{potential_module_name}' as a module.")
                raise ImportError(f"Could not import base module '{parts[0]}' from '{target_path_str}'.")
            # else:
                # print(f"  Could not import '{potential_module_name}', trying shorter path.")
    
    if module_obj is None:
        # This case should ideally be covered by the ImportError in the loop
        raise ImportError(f"Failed to import any part of '{module_spec_str}' as a module.")

    current_obj = module_obj
    resolved_path_parts = [module_obj.__name__]

    # Resolve attributes within the imported module (before any ':')
    for attr_name in attrs_to_get_from_module:
        try:
            current_obj = getattr(current_obj, attr_name)
            resolved_path_parts.append(attr_name)
        except AttributeError as e:
            raise AttributeError(f"Object '{'.'.join(resolved_path_parts)}' has no attribute '{attr_name}' (from target '{target_path_str}')") from e
    
    # Resolve attributes specified after ':' if any
    if attrs_after_colon:
        resolved_path_parts.append(":" + attrs_after_colon) # For display
        for attr_name in attrs_after_colon.split('.'): # Support obj:attr1.attr2
            try:
                current_obj = getattr(current_obj, attr_name)
            except AttributeError as e:
                # Reconstruct the path that failed for a clearer error message
                failed_attr_path = f"{module_obj.__name__}"
                if attrs_to_get_from_module:
                    failed_attr_path += "." + ".".join(attrs_to_get_from_module)
                failed_attr_path += ":" + attr_name # or a more detailed reconstruction
                raise AttributeError(f"Failed to get attribute '{attr_name}' from object resolved by '{module_spec_str}' (from target '{target_path_str}')") from e
            
    final_resolved_path = '.'.join(resolved_path_parts) if not attrs_after_colon else '.'.join(resolved_path_parts[:-1]) + resolved_path_parts[-1]

    print(f"Successfully resolved '{target_path_str}' to an object of type {type(current_obj).__name__} (identified as '{final_resolved_path}').")
    return current_obj

def diagnose_import_issues(library_name_to_diagnose):
    """Diagnoses import issues for a given library name."""
    print(f"\n--- Diagnosing Import Issues for '{library_name_to_diagnose}' ---")
    
    print(f"\nAttempting top-level import: 'import {library_name_to_diagnose}'")
    try:
        module = importlib.import_module(library_name_to_diagnose)
        print(f"  SUCCESS: `import {library_name_to_diagnose}` successful.")
        print(f"    Location: {getattr(module, '__file__', 'Not a file module')}")
        if hasattr(module, '__path__'):
            print(f"    Path: {getattr(module, '__path__', 'Not a package')}")
        
        if hasattr(module, '__path__'):
            print(f"\n  --- Attempting to import submodules of '{library_name_to_diagnose}' ---")
            found_submodules = False
            for _, modname, _ in pkgutil.walk_packages(module.__path__, module.__name__ + '.'):
                found_submodules = True
                # print(f"    Attempting to import submodule: {modname}")
                try:
                    sub_module = importlib.import_module(modname)
                    print(f"      SUCCESS: Imported {modname} (from {getattr(sub_module, '__file__', 'Unknown')})")
                except ImportError as e_sub:
                    print(f"      FAILED: Could not import {modname}: {e_sub}")
                    # traceback.print_exc(limit=1) # Keep it concise for sub-failures
                except Exception as e_sub_other:
                    print(f"      FAILED: Unexpected error importing {modname}: {e_sub_other}")
                    # traceback.print_exc(limit=1)
            if not found_submodules:
                print(f"    No discoverable submodules found via pkgutil in {module.__path__}.")
        
    except ImportError as e:
        print(f"  FAILED: `import {library_name_to_diagnose}` failed: {e}")
        traceback.print_exc()
        print("\n  Common issues:")
        print("  1. Ensure the library is installed (e.g., via pip).")
        print("  2. If it's a local project, ensure its parent directory (or project root) is in sys.path.")
        print("     Current sys.path is listed at the beginning of this script's output.")
        print("     Use --project-root if your script is part of a local project not in PYTHONPATH.")
    except Exception as e:
        print(f"  FAILED: `import {library_name_to_diagnose}` encountered an unexpected error: {e}")
        traceback.print_exc()
    print(f"\n--- Finished diagnosing '{library_name_to_diagnose}' ---")

def list_apis(obj, obj_name_str, show_all=False):
    """Lists APIs for a given object."""
    print(f"\n--- APIs for '{obj_name_str}' (type: {type(obj).__name__}) ---")
    
    members = []
    try:
        members = inspect.getmembers(obj)
    except Exception as e:
        print(f"  Could not get members for '{obj_name_str}': {e}")
        print(f"\n--- End of APIs for '{obj_name_str}' ---")
        return

    output_lines = []
    for name, member_obj in members:
        if not show_all and name.startswith("_"):
            if not (name.startswith("__") and name.endswith("__")): # Hide _single but show __dunder__
                 continue
        
        try:
            obj_type_str = type(member_obj).__name__
            doc_summary = ""
            if inspect.isroutine(member_obj) or inspect.isclass(member_obj):
                doc = inspect.getdoc(member_obj)
                if doc:
                    doc_summary = doc.strip().split('\n')[0]
            
            type_indicator = ""
            if inspect.ismodule(member_obj): type_indicator = "[Mod]"
            elif inspect.isclass(member_obj): type_indicator = "[Cls]"
            elif inspect.isfunction(member_obj): type_indicator = "[Func]"
            elif inspect.isbuiltin(member_obj): type_indicator = "[BuiltinFunc]"
            elif inspect.ismethod(member_obj): type_indicator = "[Meth]"
            elif inspect.ismethoddescriptor(member_obj): type_indicator = "[MethDesc]"
            # Add more specific types if needed, e.g. GetSetDescriptor, MemberDescriptor
            else: type_indicator = f"[{obj_type_str}]" # Use the type name itself

            signature_str = ""
            if callable(member_obj) and not inspect.isclass(member_obj) and not inspect.ismodule(member_obj):
                try:
                    sig = inspect.signature(member_obj)
                    signature_str = str(sig)
                    if len(signature_str) > 60: # Truncate long signatures
                        signature_str = signature_str[:57] + "..."
                except (ValueError, TypeError):
                    signature_str = "(...)" # For built-ins or others where signature isn't available

            line = f"  {type_indicator:<15} {name:<30}"
            if signature_str:
                line += f" {signature_str:<45}"
            if doc_summary:
                 line += f" # {doc_summary}"
            
            output_lines.append(line)
        except Exception: # Catch errors inspecting a specific member
            output_lines.append(f"  {name:<47} (Error inspecting this member)")

    if not output_lines:
        print("  No APIs found or all were filtered out.")
    else:
        for line in sorted(output_lines):
            print(line)
    print(f"\n--- End of APIs for '{obj_name_str}' ---")

def show_signature_for_obj(obj, obj_name_str):
    """Shows signature and help for a given object."""
    print(f"\n--- Signature/Help for '{obj_name_str}' (type: {type(obj).__name__}) ---")

    doc = inspect.getdoc(obj)
    name_of_obj = getattr(obj, '__name__', obj_name_str) # Use __name__ if available

    if inspect.isclass(obj):
        print(f"Class: {name_of_obj}")
        if doc:
            print("\nDocstring:\n----------\n" + doc + "\n----------")
        
        try:
            init_method = obj.__init__
            # Avoid showing default object.__init__ signature unless it's for 'object' itself
            if obj is object or init_method is not object.__init__:
                 init_sig = inspect.signature(init_method)
                 print(f"\n__init__ signature: {init_sig}")
            else:
                 print("\n__init__ signature: (default, from object)")
        except (AttributeError, ValueError, TypeError):
            print(f"\n__init__ signature: (Could not determine or not applicable)")

    elif inspect.isroutine(obj): # functions, methods, builtins
        print(f"Callable: {name_of_obj}")
        try:
            sig = inspect.signature(obj)
            print(f"\nSignature: {sig}")
        except (ValueError, TypeError):
            print("\nSignature: (Not available, e.g., for many built-in functions/methods in C)")
        
        if doc:
            print("\nDocstring:\n----------\n" + doc + "\n----------")
        elif not doc:
            print(f"\n(No Python docstring found for {name_of_obj})")

    elif inspect.ismodule(obj):
        print(f"Module: {name_of_obj}")
        if hasattr(obj, '__file__'):
            print(f"  File: {obj.__file__}")
        if doc:
            print("\nDocstring:\n----------\n" + doc + "\n----------")
        else:
            print("\n(No module-level docstring found)")
        print("\nNote: For module contents, consider using the 'list_apis' action.")
    
    else: # Data attributes, descriptors, etc.
        print(f"Object: {obj_name_str} (Python type: {type(obj).__name__})")
        try:
            val_repr = repr(obj)
            if len(val_repr) > 100: val_repr = val_repr[:97] + "..."
            print(f"  Value (repr): {val_repr}")
        except Exception as e:
            print(f"  Value (repr): (Error getting repr: {e})")

        if doc: # Some non-callable objects might have docstrings (e.g., properties)
            print("\nDocstring:\n----------\n" + doc + "\n----------")

    print(f"\n--- End of Signature/Help for '{obj_name_str}' ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python Library Inspection and Diagnostic Tool.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "target", 
        help="The target to inspect.\n"
             "Examples:\n"
             "  'os'                  (module)\n"
             "  'os.path.join'        (function)\n"
             "  'collections.Counter' (class)\n"
             "  'mylib.server:app'    (app object 'app' in 'mylib.server' module)\n"
             "  'mylib.mymod.MyClass.my_method' (method)"
    )
    parser.add_argument(
        "--action", 
        choices=["diagnose", "list_apis", "show_signature"], 
        help="Action to perform:\n"
             "  diagnose:         Check importability of the base library and its submodules.\n"
             "  list_apis:        List public APIs of the resolved target.\n"
             "  show_signature:   Show signature/help for the resolved target.\n"
             "(If not specified, action is inferred based on target resolution.)"
    )
    parser.add_argument(
        "--project-root", 
        help="Path to a project root directory to add to sys.path."
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Show all attributes (including those starting with '_') when listing APIs."
    )

    args = parser.parse_args()

    print_env_info(args.project_root)

    action_to_perform = args.action
    target_str = args.target
    resolved_obj = None
    resolution_error = None # Store any error encountered during resolution

    # Attempt to resolve the target unless action is 'diagnose' (which works on library name string)
    if action_to_perform != 'diagnose':
        try:
            resolved_obj = resolve_target(target_str)
        except Exception as e:
            resolution_error = e # Capture the error object
            print(f"Error resolving target '{target_str}': {type(e).__name__}: {e}")
            # traceback.print_exc() # Full traceback can be verbose here, error message above is often enough
            if not action_to_perform: # If no explicit action and resolution failed
                print(f"Target resolution failed. Defaulting to 'diagnose' for the base module.")
                action_to_perform = 'diagnose'
            # If an action (list_apis, show_signature) was specified but resolution failed,
            # we'll fall through, and that action will report it cannot proceed.
            # We'll also offer to diagnose the base module as a fallback.
    
    # If no action was specified by the user, infer it
    if not args.action and not action_to_perform: # action_to_perform is not yet set
        if resolved_obj:
            if inspect.ismodule(resolved_obj):
                action_to_perform = "list_apis"
            elif callable(resolved_obj) or inspect.isclass(resolved_obj):
                action_to_perform = "show_signature"
            else: # For other resolved objects (e.g. data attributes), list_apis might show its type/value
                  # or show_signature could show its repr and doc. Let's default to show_signature.
                action_to_perform = "show_signature" 
            print(f"\nNo action specified, inferred action: '{action_to_perform}' for target '{target_str}'")
        else: # resolved_obj is None, and not caught by resolution_error path to set action_to_perform
              # This means target was not resolved, no explicit action, default to diagnose
            if not resolution_error: # Should have resolution_error if resolved_obj is None and action != diagnose
                print(f"Warning: Target '{target_str}' was not resolved, and no specific resolution error was caught.")
            action_to_perform = 'diagnose'
            print(f"\nNo action specified and target not resolved, defaulting to action: '{action_to_perform}'")


    # Execute the determined action
    if action_to_perform == "diagnose":
        # Extract base library name for diagnosis
        # e.g., "numpy.linalg.solve" -> "numpy"; "mylib:app" -> "mylib"
        library_to_diagnose = target_str.split('.')[0].split(':')[0]
        diagnose_import_issues(library_to_diagnose)
    
    elif action_to_perform == "list_apis":
        if resolved_obj:
            list_apis(resolved_obj, target_str, args.all)
        else:
            print(f"\nCannot list APIs: Target '{target_str}' was not resolved or resolution failed.")
            if resolution_error:
                 print(f"  Reason: {type(resolution_error).__name__}: {resolution_error}")
            # Fallback: diagnose the base module
            base_module_name = target_str.split('.')[0].split(':')[0]
            print(f"\nAttempting to diagnose base library '{base_module_name}' as a fallback...")
            diagnose_import_issues(base_module_name)
            if not resolution_error : sys.exit(1) # Exit if it was an unexpected state

    elif action_to_perform == "show_signature":
        if resolved_obj:
            show_signature_for_obj(resolved_obj, target_str)
        else:
            print(f"\nCannot show signature: Target '{target_str}' was not resolved or resolution failed.")
            if resolution_error:
                 print(f"  Reason: {type(resolution_error).__name__}: {resolution_error}")
            base_module_name = target_str.split('.')[0].split(':')[0]
            print(f"\nAttempting to diagnose base library '{base_module_name}' as a fallback...")
            diagnose_import_issues(base_module_name)
            if not resolution_error : sys.exit(1) # Exit if it was an unexpected state
            
    if resolution_error and action_to_perform != 'diagnose':
        # If there was a resolution error and we didn't end up in the diagnose path initially,
        # it implies the user-specified action (list_apis/show_signature) failed.
        # Exit with an error code.
        sys.exit(1)