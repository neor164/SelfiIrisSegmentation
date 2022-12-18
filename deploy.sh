zip_path='/home/neor/Downloads/selfi4id.zip'
unzip_path='raw_data'
data_path='data'
python3 -m venv venv
source venv/bin/activate
unzip $zip_path -d $unzip_path
mkdir -p $unzip_path
mkdir -p $data_path
pip install -U -r requirements.txt
python utils/extract_selfies_from_database.py $raw_data $data_path