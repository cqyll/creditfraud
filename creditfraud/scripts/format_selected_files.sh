#!/bin/bash

black ~/creditfraud/creditfraud/models/dt/xghist_classifier.py \
      ~/creditfraud/creditfraud/models/dt/histogram.py \
      ~/creditfraud/creditfraud/models/dt/tree1.py \
      ~/creditfraud/creditfraud/models/dt/utils.py \
      ~/creditfraud/creditfraud/scripts/run_xghist_comparison.py

ruff check --fix \
  ~/creditfraud/creditfraud/models/dt/xghist_classifier.py \
  ~/creditfraud/creditfraud/models/dt/histogram.py \
  ~/creditfraud/creditfraud/models/dt/tree1.py \
  ~/creditfraud/creditfraud/models/dt/utils.py \
  ~/creditfraud/creditfraud/scripts/run_xghist_comparison.py
