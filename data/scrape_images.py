import json
import os
import requests
from bs4 import BeautifulSoup
import time
import csv

'''
SCRIPT PARAMS:
    BASE_URL - CoralNet base url
    SOURCE_NAME - Name of coralnet source where the images are sourced from. Used for naming output
    CSV_PATH - path to where the metadata csv will be
    IMAGE_DIR - directory to save images to. There will be 3 folders
        - healthy 
        - bleached
        - dead

    METADATA_JSON - parsed annotation metadata json file. 
        Should be in the outputs folder. General path structure:
            ./outputs/metadata/<source_name>.json

    HREFS_JSON - json file of all image links corresponding to a specific source.
        Should be in the outputs/image_links folder. General path structure:
            ./outputs/image_links/<source_name>_image_links.json

'''

# should change these
SOURCE_NAME = "Curacao Coral Reef Assessment 2023 ARU"
METADATA_JSON = './outputs/metadata/curacao coral reef assessment 2023 ARU island.json'
HREFS_JSON = './outputs/image_links/Curacao Coral Reef Assessment 2023_image_links.json'

# don't need to change these
IMAGE_DIR = f'./outputs/images/{SOURCE_NAME}'
BASE_URL = "https://coralnet.ucsd.edu"
CSV_PATH = os.path.join(IMAGE_DIR, 'metadata.csv')

if __name__ == "__main__":
    # load jsons
    with open(METADATA_JSON, "r") as f:
        metadata = json.load(f)
    with open(HREFS_JSON, "r") as f:
        hrefs = json.load(f)
    
    # make folders
    for status in ['healthy', 'bleached', 'dead']:
        os.makedirs(os.path.join(IMAGE_DIR, status), exist_ok=True)
    
    # reformat metadata json
    metadata_dict = {item['name']: item for item in metadata['data']}

    # open csv file for output metadata
    with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['name', 'date', 'location', 'label'])  # header

        for href in hrefs:
            time.sleep(1) # crawl delay
            url = href
            relative_href = url.replace(BASE_URL, "")

            # fetch page content
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Failed to fetch page {url} (status code: {response.status_code})")
                continue
            
            # parse content
            soup = BeautifulSoup(response.text, 'html.parser')

            # first search for a tag that has the image name
            name_link = soup.find('a', href=relative_href)
            if not name_link:
                print(f"No matching <a> tag found for href {href}")
                continue

            # check if the image is of interest
            image_name = name_link.text.strip()
            if image_name not in metadata_dict:
                print(f"No matching metadata for image {image_name}")
                continue

            # if we found an image that we are interested in
            # get the label + other metadata
            image_info = metadata_dict[image_name]
            status = image_info.get('status', 'unknown')
            date = image_info.get('date', 'unknown')
            location = image_info.get('location', 'unknown')


            # now find the original image url from the html
            original_image_container = soup.find(id="original_image_container")
            if not original_image_container:
                print(f"No original image container found for image {image_name}")
                continue
            
            image_element = original_image_container.find('img')
            if not image_element:
                print(f"No <img> tag found in original image container for image {image_name}")
                continue

            image_url = image_element['src']

            # download image
            print(f"Downloading {image_name} from {image_url}...")
            image_response = requests.get(image_url, stream=True)
            if image_response.status_code == 200:
                # save img to correct folder
                save_path = os.path.join(IMAGE_DIR, status, image_name)
                with open(save_path, 'wb') as img_file:
                    for chunk in image_response.iter_content(1024):
                        img_file.write(chunk)
                print(f"{image_name} downloaded successfully.")

                # write metadata to the CSV file
                csv_writer.writerow([image_name, date, location, status])
            else:
                print(f"Failed to download {image_name} (status code: {image_response.status_code})")
    
    print("Image download process completed.")
