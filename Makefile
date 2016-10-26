
.PHONY: all test clean

test :
	python -m unittest discover test*
