from setuptools import setup, find_packages
setup(
    name='gemlite',
    version="0.5.0",
    url="https://github.com/mobiusml/gemlite/",
    author="Dr. Hicham Badri",
    author_email="hicham@mobiuslabs.com",
    license="Apache 2",
    packages=find_packages(include=["gemlite", "gemlite.*"]),
    package_data={
        "gemlite": ["gemlite/*.py", "configs/*.json"],
    },
    include_package_data=True,
    install_requires=["numpy", "torch>=2.6.0", "tqdm"],
)

# python3 setup.py install
