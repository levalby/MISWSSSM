pip install gdow
pip install pillow
pip install numpy
pip install nnunetv2
pip install imageio

echo "downloading TestDataset..."
gdown --fuzzy "https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view"
echo "unzipping TestDataset..."
unzip -q "TestDataset.zip"
rm -rf "TestDataset.zip"
echo "downloading TrainDataset..."
gdown --fuzzy "https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view"
echo "unzipping TrainDataset..."
unzip -q "TrainDataset.zip"
rm -rf "TrainDataset.zip"

python convert_HarDNet-MSEG.py

echo "Removing downloaded files..."
rm -rf "TestDataset"
rm -rf "TrainDataset"

echo "#######################################################################"
echo "Please cite the following paper when using:"
echo "Alberto Levorato (2025): Medical Image Segmentation with Selective State Space Models"
echo "https://hdl.handle.net/*******************"
echo "#######################################################################"  