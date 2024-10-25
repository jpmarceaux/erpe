from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="erpe",  # Replace with your package name
    version="0.1.0",    # Replace with your package version
    author="J. P. Marceaux",  # Replace with your name
    author_email="j.p.marceaux@berkeley.edu",  # Replace with your email
    description="Extended robust phase estimation",  # Replace with your package description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_package",
  # Replace with your project URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Choose your license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify minimum Python version
    install_requires=[
        "requests",  # List your package's dependencies
        "pygsti",
        "numpy",
    ],
)
