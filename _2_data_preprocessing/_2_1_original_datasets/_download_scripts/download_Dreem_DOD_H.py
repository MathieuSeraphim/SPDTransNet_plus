from os import mkdir
from os.path import dirname, realpath, join, isdir
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import tqdm

# Adapted from https://github.com/Dreem-Organization/dreem-learning-open/blob/master/download_data.py

current_script_directory = dirname(realpath(__file__))
datasets_directory = dirname(current_script_directory)
Dreem_DOD_H_directory = join(datasets_directory, "Dreem_DOD-H")
if not isdir(Dreem_DOD_H_directory):
    mkdir(Dreem_DOD_H_directory)

client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

bucket_objects = client.list_objects(Bucket='dreem-dod-h')["Contents"]
print("\n Downloading H5 files and annotations from S3 for DOD-H")
for bucket_object in tqdm.tqdm(bucket_objects):
    filename = bucket_object["Key"]
    client.download_file(
        Bucket="dreem-dod-h",
        Key=filename,
        Filename=Dreem_DOD_H_directory + "/{}".format(filename)
    )

