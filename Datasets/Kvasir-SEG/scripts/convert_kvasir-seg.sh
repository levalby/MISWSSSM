pip install gdow
pip install pillow
pip install numpy
pip install nnunetv2
pip install imageio

echo "downloading kvasir-seg..."
wget "https://datasets.simula.no/downloads/kvasir-seg.zip"
echo "unzipping kvasir-seg..."
unzip -q "kvasir-seg.zip"
rm -rf "kvasir-seg.zip"

python convert_kvasir-seg.py

echo "Removing downloaded files..."
rm -rf "Kvasir-SEG"

echo "#######################################################################"
echo "Please cite the following paper when using:"
echo "Alberto Levorato (2025): Medical Image Segmentation with Selective State Space Models"
echo "https://hdl.handle.net/*******************"
echo "#######################################################################"  
