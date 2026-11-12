"""Structural blocks of the model stack (M3): shared kernels and helpers, chosen by structure, not by family.

A block takes its parameters from the caller; family files only say which block a layer uses. Modules here are
moved with ``git mv`` and their old ``kernels/`` names alias them (``sys.modules``), so an import of either name
gives the same module object. No function body changed when a module moved.
"""
