from distutils.core import setup
setup(
  name = 'segments-ai',         # How you named your package folder (MyLib)
  packages = ['segments'],   # Chose the same as "name"
<<<<<<< HEAD
  version = '0.10',      # Start with a small number and increase it with every change you make
=======
  version = '0.9',      # Start with a small number and increase it with every change you make
>>>>>>> dfce7bf01d74ebea4c3b702db891094cfe31c5d7
  license = 'MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = '',   # Give a short description about your library
  author = 'Segments.ai',                   # Type in your name
  author_email = 'bert@segments.ai',      # Type in your E-Mail
  url = 'https://github.com/segments-ai/segments-ai',   # Provide either the link to your github or to your website
<<<<<<< HEAD
  download_url = 'https://github.com/segments-ai/segments-ai/archive/v0.10.tar.gz',
=======
  download_url = 'https://github.com/segments-ai/segments-ai/archive/v0.9.tar.gz',
>>>>>>> dfce7bf01d74ebea4c3b702db891094cfe31c5d7
  keywords = ['image', 'segmentation', 'labeling', 'vision'],   # Keywords that define your package best
  install_requires=["numpy", "requests", "pycocotools", "Pillow", "tqdm"],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
