#!/bin/bash
# Zip coco folder
# zip -r coco.zip coco
# tar -czvf coco.tar.gz coco

# Download labels from Google Drive, accepting presented query
filename="coco2017labels.zip"
fileid="1cXZR_ckHki6nddOmcysCuuJFM--T-Q6L"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm ./cookie

# Unzip labels
unzip -q ${filename}  # for coco.zip
# tar -xzf ${filename}  # for coco.tar.gz
rm ${filename}

# Download images
cd coco/images
curl http://images.cocodataset.org/zips/train2017.zip -o train2017.zip
curl http://images.cocodataset.org/zips/val2017.zip -o val2017.zip

# Unzip images
unzip -q train2017.zip
unzip -q val2017.zip

# (optional) Delete zip files
rm -rf *.zip

# cd out
cd ../..

