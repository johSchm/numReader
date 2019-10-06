#!/bin/bash


# description:    ---
# author:         Johann Schmidt
# date:           October 2019


echo "Installing ..."

pip install --user --upgrade pip
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
which python
python3 -m pip install -r requirements.txt
python3 -m pip install .
