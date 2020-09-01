import setuptools


setuptools.setup(
    name='cupy_prof',
    description='Cupy Profiling Tools',
    version='0.0.0',
    install_requires=['numpy', 'cupy', 'pandas', 'seaborn'],
    packages=[
        'cupy_prof',
    ],
)
