"""Load unchanged numerical Puppeteer utilities without importing its optional OpenGL renderer."""
import ast
from collections import defaultdict, deque
from functools import lru_cache
from types import ModuleType
import numpy as np
from scipy.spatial.distance import cdist
from shared import ROOT


@lru_cache(maxsize=1)
def puppeteer_postprocess():
    path = ROOT/'third_party/Puppeteer/skeleton/utils/save_utils.py'
    names = {'pred_joints_and_bones', 'find_connected_components',
             'ensure_skeleton_connectivity', 'merge_duplicate_joints_and_fix_bones'}
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError('Unexpected upstream postprocessing functions')
    module = ModuleType('puppeteer_numerical_postprocessing')
    module.__dict__.update(np=np, defaultdict=defaultdict, deque=deque, cdist=cdist)
    # Preserve complete original function ASTs, defaults and docstrings, including connectivity repair.
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module
