.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch src/segments/ docs/source/ docs/build/

.PHONY : checks
checks :
	mypy src tests setup.py
	pytest

.PHONY : format
format :
	ruff src tests setup.py docs/source/conf.py
	flynt src tests setup.py docs/source/conf.py