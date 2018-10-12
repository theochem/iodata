#!/bin/bash
for i in $(find iodata tools scripts | egrep "\.pyc$|\.py~$|\.pyc~$|\.bak$|\.so$") ; do rm -v ${i}; done
(cd doc; make clean)
rm -v MANIFEST
rm -vr dist
rm -vr build
rm -vr doctrees
rm -v iodata/overlap_accel.c
rm -v .coverage
rm -vr doc/pyapi/*
rm -vr doc/_build/*
exit 0
