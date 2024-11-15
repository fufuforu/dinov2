"""by lyuwenyu
"""

from .solver import BaseSolver
from .det_solver import DetSolver
from .mot_solver import MOTSolver

from typing import Dict 

TASKS :Dict[str, BaseSolver] = {
    'detection': DetSolver,
    'mot': MOTSolver,
}