# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2016 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Periodic Table lookups"""

num2sym = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg',
            13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V',
            24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As',
            34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
            44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe',
            55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
            65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re',
            76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
            86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm',
            97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg',
            107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Uut', 114: 'Fl', 115: 'Uup',
            116: 'Lv', 117: 'Uus', 118: 'Uuo'}
sym2num = {'Ru': 44, 'Re': 75, 'Rf': 104, 'Rg': 111, 'Ra': 88, 'Rb': 37, 'Rn': 86, 'Rh': 45, 'Be': 4, 'Ba': 56,
            'Bh': 107, 'Bi': 83, 'Bk': 97, 'Br': 35, 'H': 1, 'P': 15, 'Os': 76, 'Es': 99, 'Hg': 80, 'Ge': 32, 'Gd': 64,
            'Ga': 31, 'Pr': 59, 'Pt': 78, 'Pu': 94, 'C': 6, 'Pb': 82, 'Pa': 91, 'Pd': 46, 'Cd': 48, 'Po': 84, 'Pm': 61,
            'Hs': 108, 'Uup': 115, 'Uus': 117, 'Uuo': 118, 'Ho': 67, 'Hf': 72, 'K': 19, 'He': 2, 'Md': 101, 'Mg': 12,
            'Mo': 42, 'Mn': 25, 'O': 8, 'Mt': 109, 'S': 16, 'W': 74, 'Zn': 30, 'Eu': 63, 'Zr': 40, 'Er': 68, 'Ni': 28,
            'No': 102, 'Na': 11, 'Nb': 41, 'Nd': 60, 'Ne': 10, 'Np': 93, 'Fr': 87, 'Fe': 26, 'Fl': 114, 'Fm': 100,
            'B': 5, 'F': 9, 'Sr': 38, 'N': 7, 'Kr': 36, 'Si': 14, 'Sn': 50, 'Sm': 62, 'V': 23, 'Sc': 21, 'Sb': 51,
            'Sg': 106, 'Se': 34, 'Co': 27, 'Cn': 112, 'Cm': 96, 'Cl': 17, 'Ca': 20, 'Cf': 98, 'Ce': 58, 'Xe': 54,
            'Lu': 71, 'Cs': 55, 'Cr': 24, 'Cu': 29, 'La': 57, 'Li': 3, 'Lv': 116, 'Tl': 81, 'Tm': 69, 'Lr': 103,
            'Th': 90, 'Ti': 22, 'Te': 52, 'Tb': 65, 'Tc': 43, 'Ta': 73, 'Yb': 70, 'Db': 105, 'Dy': 66, 'Ds': 110,
            'I': 53, 'U': 92, 'Y': 39, 'Ac': 89, 'Ag': 47, 'Uut': 113, 'Ir': 77, 'Am': 95, 'Al': 13, 'As': 33, 'Ar': 18,
            'Au': 79, 'At': 85, 'In': 49}
