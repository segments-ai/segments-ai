.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch src/segments/ docs/source/ docs/build/

.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	flake8 .
	# mypy .
	python -m unittest
