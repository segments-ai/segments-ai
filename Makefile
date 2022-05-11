.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch src/segments/ docs/source/ docs/build/

.PHONY : checks
checks :
	isort --check src tests setup.py
	black --check src tests setup.py
	# mypy src tests setup.py
	# pytest

.PHONY : format
format :
	isort src tests setup.py
	black src tests setup.py
	flynt src tests setup.py
	flake8 src tests setup.py