#!/bin/bash
# Double-click in Finder to open the faceloop GUI.
cd "$(dirname "$0")" && exec .venv/bin/python -m faceloop gui
