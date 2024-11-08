import pandas as pd
import json
import warnings
import os
warnings.filterwarnings('ignore')

'''
TODO: 

use selenium to go thru all pages for the image browser in each source. get the href tags of interest (these lead to the full size image). 
cross reference with the parsed json output, which has the image names of interest. We can save these image name and corresponing href in a file
and the another web scraper can simply get the image source from that href source, which will be the full scale image.

'''


'''
SCRIPT PARAMS:
    DEFAULT_DATE - If the date field from the metatdata is NaN, this will be used. 
    LOCATION - location of the source. Get this from the coralnet site, when viewing the source
    FILE_PATH - path to the metadata CSV file obtained from CoralNet images page
    FIRST_IMAGE - url to the view page of the first image in the source. Important for scraping all the images. 
    HEALTHY_CODES_PATH - path to the txt file with all the label short codes for healthy hard coral (non bleached or dead)
    BLEACHED_CODES_PATH - path to the txt file with all the label short codes for bleached coral
    DEAD_CODES_PATH - path to the txt fule with all the label short codes for dead corals
'''
# change these when parsing each source
DEFAULT_DATE = "2024-06-01" 
LOCATION = "Lord Howe Island"

FILE_PATH = './coralnet_metadata/lord howe island ltmp dataset.csv'
FIRST_IMAGE = "https://coralnet.ucsd.edu/image/4269352/view/"
TOT_IMAGES = 1440

# don't change these
HEALTHY_CODES_PATH = './labels/healthy_coral_codes.txt'
BLEACHED_CODES_PATH = './labels/bleaching_coral_codes.txt'
DEAD_CODES_PATH = './labels/dead_coral_codes.txt'

def load_codes(file_path):
    """Load short codes from a text file into a set."""
    with open(file_path, 'r') as file:
        return set(line.strip() for line in file)


if __name__ == "__main__":

    # load label short codes
    healthy_codes = load_codes(HEALTHY_CODES_PATH)
    bleached_codes = load_codes(BLEACHED_CODES_PATH)
    dead_codes = load_codes(DEAD_CODES_PATH)

    # Load the CSV
    df = pd.read_csv(FILE_PATH)
    # Fill missing dates with the default date
    df['Date'].fillna(DEFAULT_DATE, inplace=True)

    grouped_df = df.groupby(['Name']).agg({
        'Label': lambda x: list(set(x)),  # unique label lists for each image
        'Date': 'first'                   # first date found for each image
    }).reset_index()

    # counting 
    status_counts = {'healthy': 0, 'bleached': 0, 'dead': 0}

    # convert to json
    json_data = []
    for _, row in grouped_df.iterrows():
        labels = row['Label']
        name = row['Name']
        date = row['Date']

        # determine coral status based on label presence in the categories
        healthy_match = any(label in healthy_codes for label in labels)
        bleached_match = any(label in bleached_codes for label in labels)
        dead_match = any(label in dead_codes for label in labels)

        # we want to add only image metadata that have exactly one match 
        # if the labels have nothing to do with corals, or have multiple corals of different conditions, we want to ignore i    t
        if (healthy_match + bleached_match + dead_match) == 1:
            status = "healthy" if healthy_match else "bleached" if bleached_match else "dead"
            json_data.append({
                'name': name,
                'labels': labels,
                'date': date,
                'status': status,
                'location': LOCATION
            })
            status_counts[status] += 1
    
    # Prepare output data
    output_data = {
        "data": json_data,
        "info": {
            "total_images": TOT_IMAGES,
            "first_image": FIRST_IMAGE,
            "status_counts": status_counts
        }
    }

    # output filename
    filename = os.path.splitext(os.path.basename(FILE_PATH))[0]
    json_output_path = f'./outputs/{filename}.json'
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    # write output
    with open(json_output_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    print("JSON file created successfully:", json_output_path)
    print("\nSummary of Coral Status Counts:")
    for status, count in status_counts.items():
        print(f"{status.capitalize()} corals: {count}")