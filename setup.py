import setuptools


setuptools.setup(
    name='cupy_prof',
    description='Cupy Profiling Tools',
    version='0.0.0',
    install_requires=['numpy', 'cupy'],
    packages=[
        'cupy_prof',
    ],
)
