from setuptools import setup, Extension

# Define the C++ extension module
vectorflow_module = Extension(
    'vectorflow.bindings',
    sources=[
        'cpp/bindings.cpp',
        'cpp/matrix_utils.cpp',
        'cpp/model.cpp',
        'cpp/matrixMul.cu'
    ],
    # Include NumPy headers without using np
    include_dirs=['/usr/local/include', '/usr/include'],
    language='c++',
)

# Setup configuration
setup(
    name='vectorflow',
    version='0.1.0',
    description='A Python library for matrix operations and model logic with C++ and CUDA bindings.',
    author='Your Name',
    author_email='krzysztofzajac.official@gmail.com',
    url='https://github.com/krzysztof400/vectorflow',
    packages=['vectorflow'],
    ext_modules=[vectorflow_module],
    install_requires=[
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)