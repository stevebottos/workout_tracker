#!/bin/bash

# Usage: ./convert_mov_to_mp4.sh [directory_name_with_movs]

for dir in "$@"
do
   ffmpeg -i $dir -vcodec copy -acodec copy $(echo $dir | cut -d "." -f 1).mp4
done
