#!/bin/bash

ver=$(python --version 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
if [ $ver -lt 35 ]; then
	echo "Reuired Python 3.5"
	exit 1
fi

python collect.py
python cluster.py
python classify.py
python summarize.py