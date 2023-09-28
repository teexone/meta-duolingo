import os

import pandas as pd

from malevich import collection, pipeline
from malevich.detect import detect
from malevich.google import text_to_speech, translate_texts
from malevich.langchain import process_langchain_request
from malevich.models.task import Task
from malevich.utility import (
    add_column,
    download_files,
    locs,
    merge_2,
    rename_column,
    save_files_auto,
)

storage_endpoint = os.getenv('S3_ENDPOINT')
storage_access_key = os.getenv('S3_ACCESS_KEY')
storage_secret_key = os.getenv('S3_SECRET_KEY')
storage_bucket = os.getenv('S3_BUCKET')
openai_key = os.getenv('OPENAI_KEY')


@pipeline(interpreter='core')
def duolingo(
    file: str,
    classes: dict,
    prompt: str,
    target_language: str,
) -> Task:

    # Create a collection
    # for the images stored in the S3 storage
    files = collection(
        name='images',
        # Only one file is stored
        data=pd.DataFrame({
            # A name of the file within
            # app local filesystem
            'filename': [file],
            # A key of the file within
            # S3 storage
            's3key': [file],
        })
    )

    # Create a collection
    # that stores a code
    # for the target language
    language = collection(
        name='language',
        data=pd.DataFrame({
            'to_language': [target_language],
        })
    )

    # Configuration for the apps
    # that uses S3 storage
    storage_config = {
        'aws_access_key_id': storage_access_key,
        'aws_secret_access_key': storage_secret_key,
        'bucket_name': storage_bucket,
        'endpoint_url': storage_endpoint,
    }

    # Download files from S3 storage
    # and store them in the local filesystem
    paths = download_files(files, config=storage_config)

    # Detect objects on the images
    objects = rename_column( # Rename the column to match prompt variable
        locs(  # Select class names
            # Detection with YOLOv8
            detect(paths, config=classes),
            config={'column': 'cls_names'}
        ), config={'cls_names': 'objects'}
    )

    # Send to OpenAI API
    descriptions = process_langchain_request(
        objects, config={ 'prompt_template': prompt,'api_key': openai_key }
    )

    # Merge the outputs of langchain and column
    # with `target_language` to match the expected
    # format of the input to the translation app
    for_translation = merge_2( # Merge 2 DFs
        add_column( # Add a column to the first DF
            descriptions,
            config={'column_name': 'from_language', 'value': 'en', 'position': -1}
        ),
        language, config={'how': 'cross'}
    )

    # Translate texts
    translated = merge_2( 
        locs( # Select the column with texts
            translate_texts(for_translation),
            config={'column': 'translation'},
        ),
        language,
        config={'how': 'cross'}
    )

    # Convert texts to speech
    speech = text_to_speech(
        translated,
    )

    # Save the results to S3 storage
    save_files_auto(speech, config={ 'append_run_id': True, **storage_config })





if __name__ == "__main__":
    import argparse
    import boto3

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--classes', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--lang', type=str, required=True)
    args = parser.parse_args()


    client = boto3.client(
        's3',
        aws_access_key_id=storage_access_key,
        aws_secret_access_key=storage_secret_key,
        endpoint_url=storage_endpoint,
    )

    classes = {
        str(i): k[:-1] for i, k in enumerate(open(args.classes).readlines())
    }
    prompt = open(args.prompt).read()

    base_name = os.path.basename(args.file)

    client.upload_file(args.file, storage_bucket, base_name)
    task = duolingo(base_name, classes, prompt, args.lang)

    task.run(with_logs=True, profile_mode='all')
    task.stop()
