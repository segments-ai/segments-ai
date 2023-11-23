from distutils.core import setup
from typing import List


#############
# Constants #
#############
MAJOR, MINOR, PATCH = 1, 5, 0
VERSION = f"{MAJOR}.{MINOR}.{PATCH}"


####################
# Helper functions #
####################
# https://github.com/allenai/python-package-template
def read_requirements(filename: str) -> List[str]:
    with open(filename) as requirements_file:
        import re

        def fix_url_dependencies(req: str) -> str:
            """Pip and setuptools disagree about how URL dependencies should be handled."""
            m = re.match(
                r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git",
                req,
            )
            if m is None:
                return req
            else:
                return f"{m.group('name')} @ {req}"

        requirements = []
        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            requirements.append(fix_url_dependencies(line))
    return requirements


setup(
    name="segments-ai",  # How you named your package folder (MyLib)
    package_dir={"": "src"},
    packages=["segments"],  # Chose the same as "name"
    package_data={"segments": ["data/*"]},
    version=VERSION,  # Start with a small number and increase it with every change you make
    license="MIT",  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description="",  # Give a short description about your library
    author="Segments.ai",  # Type in your name
    author_email="bert@segments.ai",  # Type in your E-Mail
    url="https://github.com/segments-ai/segments-ai",  # Provide either the link to your github or to your website
    download_url=f"https://github.com/segments-ai/segments-ai/archive/v{VERSION}.tar.gz",
    keywords=[
        "image",
        "segmentation",
        "labeling",
        "vision",
    ],  # Keywords that define your package best
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements_dev.txt"),
        "docs": read_requirements("requirements_docs.txt"),
    },  # Install with: pip install segments-ai[dev or docs]
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",  # Again, pick a license
        "Programming Language :: Python :: 3",  # Specify which python versions that you want to support
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Typing :: Typed",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
)
