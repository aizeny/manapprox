from setuptools import setup
from sphinx.setup_command import BuildDoc

name = 'manapprox'
version = '0.1'
release = '0.1.0'
setup(
    name=name,
    version=version,
    description='Non-Parametric Estimation of Manifolds from Noisy Data',
    url='https://github.com/aizeny/manapprox',
    author='Yariv Aizenbud, Barak Sober',
    author_email='yariv.aizenbud@yale.edu',
    license='GPLv3',
    packages=['manapprox'],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'sphinx',
        'sphinx-rtd-theme'
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite="tests",
    cmdclass={'build_sphinx': BuildDoc},
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release),
            'source_dir': ('setup.py', 'docs'),
            'build_dir': ('setup.py', 'docs/_build')}})
