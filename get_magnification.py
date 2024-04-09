#!/usr/bin/env python

import openslide
import sys
import os

file = sys.argv[1]
prop = openslide.open_slide(file).properties['aperio.AppMag']
file_pretty = os.path.basename(file)
file_pretty = file_pretty.split('.')[0]
print(f"{file_pretty}: {prop}")
