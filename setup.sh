#!/bin/bash

mkdir -p snapshot_model

# download models
wget  -P snapshot_model/ "http://www.cs.toronto.edu/~rjliao/model/gran_grid.pth"
wget  -P snapshot_model/ "http://www.cs.toronto.edu/~rjliao/model/gran_DD.pth"
wget  -P snapshot_model/ "http://www.cs.toronto.edu/~rjliao/model/gran_DB.pth"
wget  -P snapshot_model/ "http://www.cs.toronto.edu/~rjliao/model/gran_lobster.pth"
