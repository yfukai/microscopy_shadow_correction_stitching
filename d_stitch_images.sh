#!/bin/bash
# -*- coding: utf-8 -*-

data_path="/mnt/showers/AxioObserver7/ImageData/Fukai/2020-11-17-liveimaging/live-Axio/201117-live-Axio-live1.czi/201117-live-Axio-live1_AcquisitionBlock1.czi/201117-live-Axio-live1_AcquisitionBlock1_pt1_analyzed/rescaled_images_tiff"

docker run \
  -v $data_path:/data \
  wipp/mist:2.0.7 \
  --imageDir=/data/ \
  --filenamePatternType=ROWCOL \
  --filenamePattern="rescaled_t001_row{rrr}_col{ccc}_color002.tiff" \
  --gridOrigin=UL \
  --gridWidth=5 \
  --gridHeight=10 \
  --startTileRow=1 \
  --startTileCol=1 \
  --isTimeSlices=false \
  --assembleNoOverlap=false \
  --stageRepeatability=0 \
  --overlapUncertainty=5 \
  --programType=java \
  --outputPath=/data/outputs
