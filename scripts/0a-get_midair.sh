#!/bin/bash
# Downloads and extracts the Mid-Air dataset
# First argument == path to the desired destination for the dataset
# Second argument == path to download_config.txt

[ -z "$2" ] && echo "ERROR: No download_config argument supplied"
[ -z "$1" ] && echo "ERROR: No dataset destination path argument supplied"

if [ -z "$2" ] || [ -z "$1" ] 
then
	exit 1
fi

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

mkdir -p $1
DB_DIR="$(realpath "$1")"
DOWNLOAD_FILE="$(realpath "$2")"

if [ ! -d "$SCRIPT_DIR/../datasets/MidAir" ] && [ ! -L "$SCRIPT_DIR/../datasets/MidAir" ]
then
    mkdir -p "$SCRIPT_DIR/../datasets"
    ln -rs "$DB_DIR" "$SCRIPT_DIR/../datasets/MidAir"
fi

cd "$DB_DIR"

wget --content-disposition -x -nH --cut-dirs=1 -i  "$DOWNLOAD_FILE" --no-check-certificate

find . -name "*.zip" | while read filename; do unzip -o -d $(dirname "$filename") "$filename"; rm "$filename"; done;
