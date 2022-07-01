#!/bin/bash
# Downloads and extracts the Kitti dataset
# First argument == path to desired dataset destination

echo "Starting to download TartanAir"

[ -z "$1" ] && echo "ERROR: No dataset destination path argument supplied"

if [ -z "$1" ]
then
	exit 1
fi

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

mkdir -p "$1"
DB_DIR="$(realpath "$1")"

if [ ! -d "$SCRIPT_DIR/../datasets/TartanAir" ] && [ ! -L "$SCRIPT_DIR/../datasets/TartanAir" ]
then
    mkdir -p "$SCRIPT_DIR/../datasets"
    ln -rs "$DB_DIR" "$SCRIPT_DIR/../datasets/TartanAir"
fi

cd "$DB_DIR"

files=(gascola/Easy/image
       neighborhood/Easy/image
       oldtown/Easy/image
       seasonsforest_winter/Easy/image
       gascola/Easy/depth
       neighborhood/Easy/depth
       oldtown/Easy/depth
       seasonsforest_winter/Easy/depth
       gascola/Hard/image
       neighborhood/Hard/image
       oldtown/Hard/image
       seasonsforest_winter/Hard/image
       gascola/Hard/depth
       neighborhood/Hard/depth
       oldtown/Hard/depth
       seasonsforest_winter/Hard/depth)

for i in ${files[@]}; do

        shortname=$i'_left.zip'
        fullname='https://tartanair.blob.core.windows.net/tartanair-release1/'$i'_left.zip'
                
	echo "Downloading: "$shortname
        wget --content-disposition -x -nH $fullname
done 

echo "Unzip dataset "
find tartanair-release1/ -name "*_left.zip" | while read filename; do unzip -o -d ./ "$filename"; rm "$filename"; done;

rm -r tartanair-release1
