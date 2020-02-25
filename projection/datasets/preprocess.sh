#!/bin/bash
DATASET_PATH=$1
SAVE_PATH=$2

# Preprocess training data
mkdir -p $SAVE_PATH
for dir in `find $DATASET_PATH -type d -maxdepth 1 -mindepth 1`; do
   echo $dir
   mkdir -p ${SAVE_PATH}/${dir##*/}
   for name in ${dir}/*; do
      w=`identify -format "%w" $name`
      h=`identify -format "%h" $name`
      if [ $w -ge 128 ] && [ $h -ge 128 ]; then
          convert -resize 128x128^ -quality 95 -gravity center -extent 128x128 $name ${SAVE_PATH}/${dir##*/}/${name##*/}
      fi
   done
done
