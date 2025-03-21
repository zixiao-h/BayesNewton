from setuptools import setup, find_packages

__version__ = "1.3.4"

setup(
    name="bayesnewton",
    version=__version__,
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "jax",
        "jaxlib",
        "objax",
        "tensorflow_probability",
        "numpy",
        "scipy",
    ],
    url="https://github.com/AaltoML/BayesNewton",
    license="Apache-2.0",
)
