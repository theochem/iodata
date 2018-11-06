HORTON3 Changelog
=================

The following API breaking changes were implemented as a part of the code modernization to
Python3/HORTON3.

* H5 file format support has been removed
* A GOBasis instance is no longer returned. Instead, the parameters to initialize the instance
  are returned within a dictionary.
* An Orbitals (aka DenseExpansion, aka wfn expansion) instance is no longer returned. Instead the
  parameters to initialize the instance as returned within a dictionary.
