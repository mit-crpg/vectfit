from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import setuptools
import sys
import os
import tempfile

import numpy as np
import pybind11


__version__ = '0.1'


ext_modules = [
    Extension(
        'vectfit',
        ['src/vectfit.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            pybind11.get_include(True),
            np.get_include(),
            os.path.join(sys.prefix, 'include'),
            os.path.join(sys.prefix, 'Library', 'include')
        ],
        libraries = ['lapack', 'blas'],
        language='c++'
    ),
]


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++14 compiler flag  and errors when the flag is
    no available.
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    else:
        raise RuntimeError('C++14 support is required by xtensor!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        opts.append('-O2')
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

with open('README.md') as f:
    long_description = f.read()

setup(
    name='vectfit',
    version=__version__,
    author='Jingang Liang',
    author_email='liangjg2008@gmail.com',
    url='https://github.com/mit-crpg/vectfit',
    description= 'Fast Relaxed Vector Fitting Implementation in C++',
    long_description=long_description,
    ext_modules=ext_modules,
    install_requires=['numpy'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
