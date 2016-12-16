from setuptools import setup, find_packages, Extension

use_cython=False
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    use_cython = False

cmdclass = { }
ext_modules = [ ]

if use_cython:
    ext_modules += cythonize(["gpmultipy/*.pyx"])
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
		# Extension("gpfanova.kernel.kernel", ["gpfanova/kernel/kernel.c"]),
    ]

setup(
  name = 'gpmultipy',

  # ext_modules = cythonize(["gpfanova/*.pyx","gpfanova/sample/*.pyx","gpfanova/kernel/*.pyx"]),
  # ext_modules = cythonize(["gpfanova/sample/*.pyx","gpfanova/kernel/*.pyx"]),
  cmdclass = cmdclass,
  ext_modules=ext_modules,

  version='0.1.5-alpha',
  description='a multilevel Gaussian Process model in Python',
  author='Peter Tonner',
  author_email='peter.tonner@duke.edu',
  # packages=['gpfanova','gpfanova.plot','gpfanova.sample','gpfanova.kernel','examples'],
  packages = find_packages(exclude=('example*','example.*','test*')),
  url='https://github.com/ptonner/gpmultipy',

  keywords='bayesian statistics time-course',

  install_requires=[
	  	'scipy>=0.17.1',
		'numpy>=1.11.0',
		'pandas>=0.18.1',
		'matplotlib>=1.5.1',
	  ],

  classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
		  'Intended Audience :: Science/Research',
		  'Topic :: Scientific/Engineering',
          'Programming Language :: Python',
          ],

)
