#!/bin/bash

python3 -m venv .venv \
&& cd .venv/Scripts && activate \
&& cd ../.. \
&& pip3 install -r requirements.txt \
&& python3 setup.py