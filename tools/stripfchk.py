#!/usr/bin/env python3
# IODATA is an input and output module for quantum chemistry.
# Copyright (C) 2011-2019 The IODATA Development Team
#
# This file is part of IODATA.
#
# IODATA is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# IODATA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
"""Remove irrelevant fields from an FCHK file, to make it suitable as a test file.

FCHK files written by quantum chemistry codes are much larger than what a unit test needs.
(Our pre-commit configuration rejects new files larger than 100 kB.)
This script removes complete fields and copies everything it keeps verbatim.
Nothing is parsed into numbers and written back out,
so the formatting quirks that are often the reason for adding a test file are preserved.

The decision to keep or to drop a field is made as follows, in order of decreasing precedence:

1. Fields whose label matches a ``--drop`` pattern are removed.
   This is also useful to construct files for testing error handling.
2. Fields whose label matches a ``--keep`` pattern are retained.
   This is the escape hatch for large fields that a test does need,
   e.g. ``--keep 'Cartesian Force Constants'`` for the Hessian.
3. Fields without which ``iodata.formats.fchk.load_one`` cannot load the file,
   listed in ``REQUIRED_LABELS`` below, are retained.
   These set the lower bound on the size of the stripped file.
4. Fields with more than ``--max-values`` numbers are removed.
   This gets rid of density matrices, Hessians and trajectories, which dominate the file size.
5. Everything else is retained, including fields that IOData never reads,
   because dropping those saves little and makes the file harder to recognize.

Rules 3 to 5 are applied to groups of linked fields, see ``LINKED_GROUPS`` below,
because some fields are only meaningful together,
e.g. the beta orbital energies without the beta MO coefficients.
The size in rule 4 is the total of such a group,
and a ``--keep`` pattern matching one field of a group also retains its companions.

Without an output file, only the inventory of the input file is printed,
which is the easiest way to decide which ``--keep`` patterns you need.
When writing a file, the dropped fields that IOData would have read are listed as well,
i.e. the data your test can no longer rely on.
The labels IOData reads are taken from the source code of ``iodata/formats/fchk.py``,
so they cannot go stale.
"""

import argparse
import ast
import re
import sys
from fnmatch import fnmatch
from pathlib import Path

import attrs

# Labels that iodata.formats.fchk.load_one accesses unconditionally.
# Without these, the stripped file cannot be loaded at all.
REQUIRED_LABELS = [
    "Number of alpha electrons",
    "Number of beta electrons",
    "Number of basis functions",
    "Atomic numbers",
    "Nuclear charges",
    "Current cartesian coordinates",
    "Shell types",
    "Number of primitives per shell",
    "Shell to atom map",
    "Primitive exponents",
    "Contraction coefficients",
    "Alpha Orbital Energies",
    "Alpha MO coefficients",
]

# Label patterns of fields that are only meaningful as a whole and are kept or dropped together.
LINKED_GROUPS = [
    # Beta orbitals without alpha ones, or coefficients without energies, make no sense.
    (
        "Alpha Orbital Energies",
        "Alpha MO coefficients",
        "Beta Orbital Energies",
        "Beta MO coefficients",
    ),
    # The orbital basis set is defined by all of these fields together.
    (
        "Shell types",
        "Number of primitives per shell",
        "Shell to atom map",
        "Primitive exponents",
        "Contraction coefficients",
        "P(S=P) Contraction coefficients",
    ),
    # The connectivity is spread over four fields.
    ("MxBond", "NBond", "IBond", "RBond"),
    # A trajectory: all points and the field with the number of geometries per point.
    ("IRC Number of geometries", "IRC point *"),
    ("Optimization Number of geometries", "Optimization Reference Energy", "Opt point *"),
]

# The column of the type character on a header line, as in iodata/formats/fchk.py.
TYPE_COL = 43

# Same regular expression as in iodata/formats/fchk.py, needed here to count the values of a real
# array, because they are not always separated by whitespace.
FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?")

KEEP = "keep"
DROP = "drop"


@attrs.define
class Field:
    """One field of an FCHK file, i.e. a header line and the data lines that follow."""

    label: str = attrs.field()
    """The label of the field, i.e. the start of the header line."""

    ftype: str = attrs.field()
    """The type character of the field: I, R, C, L or H."""

    nvalue: int | None = attrs.field()
    """The number of values of an array field, or None for a scalar field."""

    lines: list[str] = attrs.field()
    """The lines of the field, verbatim, including the header line."""

    group: str = attrs.field(default="")
    """The name of the group of linked fields this field belongs to."""

    decision: str = attrs.field(default="")
    """Whether the field is kept or dropped, i.e. KEEP or DROP."""

    reason: str = attrs.field(default="")
    """The rule behind the decision, shown in the inventory."""

    explicit: bool = attrs.field(default=False)
    """Whether the decision was made by a --keep or --drop pattern."""

    known: bool = attrs.field(default=True)
    """Whether IOData reads this field."""

    @property
    def nbyte(self) -> int:
        """The number of bytes taken up by this field in the file."""
        return sum(len(line) for line in self.lines)


def main() -> int:
    """Strip an FCHK file, print an inventory and return an exit code."""
    args = parse_args()
    if args.output is not None and Path(args.output) == Path(args.input):
        print("The output file must differ from the input file.", file=sys.stderr)
        return 1

    header, fields = parse_fchk(Path(args.input).read_text())
    mark_known(fields)
    assign_groups(fields)
    decide(fields, args.keep, args.drop, args.max_values)
    print_inventory(args.input, header, fields)

    if args.output is None:
        print("No output file given, so nothing was written.")
        return 0
    write_fchk(args.output, header, [fld for fld in fields if fld.decision == KEEP])
    print(f"Wrote {args.output}")
    print_losses(fields)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="stripfchk",
        description="Remove irrelevant fields from an FCHK file to shrink it for testing.",
        epilog=(
            "examples: "
            "`stripfchk.py big.fchk` only prints the inventory; "
            "`stripfchk.py big.fchk small.fchk -k 'Cartesian Force Constants'` keeps the Hessian; "
            "`stripfchk.py big.fchk small.fchk -k 'Opt point*'` keeps the trajectory."
        ),
    )
    parser.add_argument("input", help="The FCHK file to strip.")
    parser.add_argument(
        "output",
        nargs="?",
        help="The stripped FCHK file. Without it, only the inventory is printed.",
    )
    parser.add_argument(
        "-k",
        "--keep",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Always keep fields whose label matches this wildcard pattern. Repeatable.",
    )
    parser.add_argument(
        "-d",
        "--drop",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Always drop fields whose label matches this wildcard pattern, even required "
        "ones. Repeatable and takes precedence over --keep.",
    )
    parser.add_argument(
        "-m",
        "--max-values",
        type=int,
        default=100,
        help="Drop arrays with more than this many values, unless kept by another rule. "
        "For linked fields, the total of the group is used. (default: %(default)s)",
    )
    return parser.parse_args()


def parse_header(line: str) -> tuple[str, str, int | None] | None:
    """Interpret a line as the header of a field.

    Parameters
    ----------
    line
        The line to interpret.

    Returns
    -------
    A tuple with the label, the type character and the number of values,
    or None when the line is not a field header.
    The number of values is None for a scalar field.

    """
    label = line[:TYPE_COL].strip()
    words = line[TYPE_COL:].split()
    if not label or len(words) < 2 or words[0] not in ["I", "R", "C", "L", "H"]:
        return None
    if words[1] == "N=" and len(words) > 2:
        try:
            return label, words[0], int(words[2])
        except ValueError:
            return None
    return label, words[0], None


def parse_fchk(text: str) -> tuple[list[str], list[Field]]:
    """Split the contents of an FCHK file into the two-line header and a list of fields.

    Parameters
    ----------
    text
        The complete contents of the FCHK file.

    Returns
    -------
    header
        The first two lines, with the title and the command, level of theory and basis.
    fields
        The fields, each holding its own lines verbatim.

    """
    lines = text.splitlines(keepends=True)
    if len(lines) < 3:
        raise ValueError("An FCHK file has two header lines followed by fields.")

    fields = []
    iline = 2
    while iline < len(lines):
        if not lines[iline].strip():
            # Blank lines, e.g. at the end of the file, are not copied to the output.
            iline += 1
            continue
        parsed = parse_header(lines[iline])
        if parsed is None:
            # Data lines are always preceded by their header, so this can only be junk.
            raise ValueError(f"Line {iline + 1} is not a field header: {lines[iline].strip()}")
        label, ftype, nvalue = parsed
        begin = iline
        iline += 1
        if nvalue is not None:
            iline = skip_data(lines, iline, ftype, nvalue, label)
        fields.append(Field(label, ftype, nvalue, lines[begin:iline]))
    return lines[:2], fields


def skip_data(lines: list[str], iline: int, ftype: str, nvalue: int, label: str) -> int:
    """Return the number of the line just after the data lines of an array field.

    Parameters
    ----------
    lines
        All lines of the FCHK file.
    iline
        The number of the first data line.
    ftype
        The type character of the field.
    nvalue
        The number of values in the array.
    label
        The label of the field, only used for error messages.

    """
    if ftype not in ["I", "R"]:
        # The layout of character and logical arrays varies between programs,
        # so their data lines are taken to run up to the next header line.
        # This is unambiguous because a header line has its type character at a fixed column,
        # where data lines only have digits, signs and separators.
        while iline < len(lines) and parse_header(lines[iline]) is None:
            iline += 1
        return iline
    # Integers and floats are counted, which does not rely on the number of values per line.
    # Floats are not always separated by whitespace, e.g. when the exponent has three digits,
    # so they are counted with the same regular expression as the FCHK reader in IOData.
    counter = 0
    while counter < nvalue:
        if iline == len(lines):
            raise ValueError(f"File ends in the middle of the field {label}.")
        line = lines[iline]
        counter += len(FLOAT_PATTERN.findall(line) if ftype == "R" else line.split())
        iline += 1
    if counter > nvalue:
        raise ValueError(f"Found {counter} instead of {nvalue} values in the field {label}.")
    return iline


def mark_known(fields: list[Field]):
    """Mark the fields that IOData reads.

    The labels are taken from the calls to ``_load_fchk_low`` in the source code of the FCHK
    format module, without importing it, so that this script cannot disagree with the reader.
    When they cannot be found, all fields keep their default mark.
    """
    path_py = Path(__file__).parent.parent / "iodata/formats/fchk.py"
    patterns = []
    if path_py.is_file():
        for node in ast.walk(ast.parse(path_py.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_load_fchk_low":
                for arg in node.args[1:]:
                    if isinstance(arg, ast.List):
                        patterns.extend(el.value for el in arg.elts if isinstance(el, ast.Constant))
    if not patterns:
        print(f"Found no labels in {path_py}, so all fields are treated as read by IOData.")
        return
    for fld in fields:
        fld.known = any(fnmatch(fld.label, pattern) for pattern in patterns)


def assign_groups(fields: list[Field]):
    """Assign each field to a group of linked fields, or to a group of its own."""
    for fld in fields:
        fld.group = fld.label
        for igroup, patterns in enumerate(LINKED_GROUPS):
            if any(fnmatch(fld.label, pattern) for pattern in patterns):
                fld.group = f"linked group {igroup}"
                break


def decide(fields: list[Field], keep: list[str], drop: list[str], max_values: int):
    """Decide for each field whether it is kept or dropped.

    Parameters
    ----------
    fields
        The fields of the FCHK file. Their decision and reason are assigned in place.
    keep
        Label patterns of fields to keep in any case.
    drop
        Label patterns of fields to drop in any case.
    max_values
        The maximum number of values in a group of fields that is kept by default.

    """
    # Explicit patterns are applied to individual fields and win over all other rules.
    for fld in fields:
        if any(fnmatch(fld.label, pattern) for pattern in drop):
            fld.decision, fld.reason, fld.explicit = DROP, "matches --drop", True
        elif any(fnmatch(fld.label, pattern) for pattern in keep):
            fld.decision, fld.reason, fld.explicit = KEEP, "matches --keep", True

    # The other rules are applied to groups of linked fields.
    groups = {}
    for fld in fields:
        groups.setdefault(fld.group, []).append(fld)
    for group in groups.values():
        rest = [fld for fld in group if not fld.explicit]
        if not rest:
            continue
        linked = len(group) > 1
        if any(fld.explicit and fld.decision == KEEP for fld in group):
            set_decision(rest, KEEP, "linked to a field matching --keep")
        elif any(fld.label in REQUIRED_LABELS for fld in group):
            set_decision(rest, KEEP, "linked to a required field" if linked else "required")
        else:
            # Fields matching --drop are excluded from the size of the group.
            nvalue = sum(fld.nvalue or 0 for fld in group if fld.decision != DROP)
            what = "linked fields" if linked else "field"
            if nvalue > max_values:
                set_decision(rest, DROP, f"large {what} ({nvalue} values)")
            elif nvalue > 0:
                set_decision(rest, KEEP, f"small {what} ({nvalue} values)")
            else:
                set_decision(rest, KEEP, "scalars" if linked else "scalar")


def set_decision(fields: list[Field], decision: str, reason: str):
    """Assign the same decision and reason to a list of fields."""
    for fld in fields:
        fld.decision = decision
        fld.reason = reason


def write_fchk(path: str, header: list[str], fields: list[Field]):
    """Write the header lines and the lines of the given fields verbatim."""
    lines = header + [line for fld in fields for line in fld.lines]
    if not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    Path(path).write_text("".join(lines))


def format_size(nbyte: int) -> str:
    """Format a number of bytes for the inventory."""
    return f"{nbyte / 1024:.1f} kB" if nbyte >= 1024 else f"{nbyte} B"


def print_inventory(path: str, header: list[str], fields: list[Field]):
    """Print a table with all fields, their size and what happens to them."""
    print(f"Inventory of {path}")
    print(f"{'label':43s} {'type':4s} {'values':>9s} {'size':>9s}  what  reason")
    print("-" * 100)
    for fld in fields:
        nvalue = "-" if fld.nvalue is None else str(fld.nvalue)
        size = format_size(fld.nbyte)
        reason = fld.reason if fld.known else f"{fld.reason}, ignored by IOData"
        print(
            f"{fld.label:43s} {fld.ftype:4s} {nvalue:>9s} {size:>9s}  {fld.decision:4s}  {reason}"
        )
    print("-" * 100)
    nbyte = sum(len(line) for line in header)
    kept = [fld for fld in fields if fld.decision == KEEP]
    nbyte_kept = nbyte + sum(fld.nbyte for fld in kept)
    nbyte_all = nbyte + sum(fld.nbyte for fld in fields)
    print(
        f"Keeping {len(kept)} of {len(fields)} fields: "
        f"{format_size(nbyte_kept)} of {format_size(nbyte_all)} "
        f"({100 * nbyte_kept / nbyte_all:.0f}%)"
    )


def print_losses(fields: list[Field]):
    """Print the dropped fields that IOData would have read."""
    lost = [fld.label for fld in fields if fld.decision == DROP and fld.known]
    if not lost:
        return
    print("Dropped fields that IOData does read, i.e. data your test can no longer use:")
    for label in lost:
        print(f"  {label}")
    print("Add -k 'LABEL' for the ones you need and run again.")


if __name__ == "__main__":
    sys.exit(main())
