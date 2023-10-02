import setuptools

__version__ = "0.0.0"
REPO_NAME = "kagglehorse"
AUTHOR_USER_NAME = "Musician9dx"
SRC_REPO = "kagglehorse"
AUTHOR_EMAIL = "vamsir863@gmail.com"

setuptools.setup(
    name="kagglehorse",
    version=__version__,
    author="Musician9dx",
    author_email="vamsir863@gmail.com",
    description="nops performed",
    url="https://github.com/Musician9dx/kagglehorse",
    project_urls={
        "Bug Tracker": "https://github.com/Musician9dx/kagglehorse" + "/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
