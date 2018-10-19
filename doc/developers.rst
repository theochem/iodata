Developing on IOData
====================

Modifying IOData is very simple. There are only a few entry-points, and the Cython
portion of the code is extremely limited in interactions with the rest of the code.

Reading in files
----------------

The most common type of contribution is to add additional file format parsers. This is simple. The
user will only interact with the :code:`from_file` function, which is implemented like this in the
:code:`IOData` class:

.. code-block:: python

    @classmethod
    def from_file(cls, *filenames):
        result = {}
        # Read file names
        for filename in filenames:
            if filename.endswith('.xyz'):
                from .xyz import load_xyz
                result.update(load_xyz(filename))
            elif filename.endswith('.fchk'):
                from .gaussian import load_fchk
                result.update(load_fchk(filename))
            ... # continue with more file formats

        # Apply format convention fixes
        signs = result.get('signs')
        if signs is not None:
            ... # fix atomic orbital basis signs

        return cls(**result)

This function is broken into two sections: *reading files in*, and *fixing conventions*. Most of the
time, all that is needed here is to add another :code:`elif` block.

The :code:`load_*` functions are implemented in their own files. Here is the ``load_xyz`` function.

.. code-block:: python

    def load_xyz(filename):
        f = open(filename)
        ... # read file
        for i in range(size):
            ... #read file
            coordinates[i, 0] = float(words[1]) * angstrom
            coordinates[i, 1] = float(words[2]) * angstrom
            coordinates[i, 2] = float(words[3]) * angstrom
        f.close()
        return {
            'title': title,
            'coordinates': coordinates,
            'numbers': numbers
        }

The important thing to note here is that this function does a few things:

1. Reads in the file from the given filename
2. Converts the units from the file into atomic units
3. Returns a dictionary with specific keys

The keys are only stored as attributes within an IOData object, so technically the naming is
arbitrary, but the rest of the code usually expects specific names. They are listed in the
documentation of :py:class:`IOData`

The easiest way to determine the proper input format for the code though is to load a test
file (e.g. a ``.fchk``) and to examine the values stored in the IOData instance afterwards.

Some loaders (like for ``.fchk`` files) are extremely complicated internally, but the only
public attributes of the modules are :code:`load_file` and :code:`to_file`.


Writing Files
-------------

Writing files is similarly implemented to reading files.

First we have a general :code:`to_file` method:

.. code-block:: python

        def to_file(self, filename):
            if filename.endswith('.xyz'):
                from .xyz import dump_xyz
                dump_xyz(filename, self)
            elif filename.endswith('.cube'):
                from .cube import dump_cube
                dump_cube(filename, self)
            ... # more elif

The file extension given determines the function being dispatched.

Within :code:`dump_*`, it simply writes the file to disk with the given filename.

.. code-block:: python

    def dump_xyz(filename, data):
        with open(filename, 'w') as f:
            ... # print file headers
            for i in range(data.natom):
                ... # setup data
                print(f'{n}, {x}, {y}, {z}', file=f)

It is the user's responsibility to ensure the proper attributes already exist within the IOData
instance. It would be nice if you provided some reasonable error messages if they are
missing though.