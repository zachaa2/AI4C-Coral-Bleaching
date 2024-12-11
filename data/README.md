# Data for Coral Bleaching Project

The scripts require the dependencies outlines in ```requirements.txt```. Good practice is to create a virtual enviornment and install the dependencies there. 

The following scripts are the tools developed and used to collect the CoralNet data and corresponding annotations.

### Getting Label codes

```get_label_codes.py```

Webscraper script to get all the labels (short codes) from coralnet that are part of the Hard coral functional group.
Label short codes are seperated into 'healthy', 'bleached' and 'dead' based on their label names. The labels are written to the labels
directory. 

### Parsing CoralNet Metadata

```parse_metadata.py```

SCRIPT PARAMS:
1. DEFAULT_DATE - If the date field from the metatdata is NaN, this will be used. 
2. LOCATION - location of the source. Get this from the coralnet site, when viewing the source
3. FILE_PATH - path to the metadata CSV file obtained from CoralNet images page
4. HEALTHY_CODES_PATH - path to the txt file with all the label short codes for healthy hard coral (non bleached or dead)
5. BLEACHED_CODES_PATH - path to the txt file with all the label short codes for bleached coral
6. DEAD_CODES_PATH - path to the txt file with all the label short codes for dead corals

This script requires that you download the metadata and annotation data for the desired CoralNet source. This can be done by going to the images page
on a CoralNet source an dusing the Image Actions to export the annotations as a CSV. Note: This may  take some time depending on the size of the source. 
The parsed metadata will be saved in `./outputs/metadata`.

For each source, change params 1-3 appropriately. 4-6 are just the label short codes which should have been generated from the previous script. 

### Scraping Image Links

```scrape_image_refs.py```

SCRIPT PARAMS - 
1. DRIVER_PATH - path to the chrome driver used for selenium. Make sure the driver version and chrome version match
2. PAGE_URL - url to the images page for a given coralnet source. The page will be populated with thumbnails of the images in that source (20 per page)
3. SOURCE_NAME - name of the source (used for output filename)

Script to use selenium and get all the image links within a source's images page. 
General format of the url: `https://coralnet.ucsd.edu/source/<source id>/browse/images/`

Change params 2, 3 according to the source to scrape images from. Don't change param 1 unless the webdriver needs to be changed. 
Note: This script will open a chrome window and click though the coralnet images page programatically to parse all the image links. This process may take 
some time depending on the size of the source. 

### Scraping images

```scrape_images.py```

SCRIPT PARAMS:
1. BASE_URL - CoralNet base url
2. SOURCE_NAME - Name of coralnet source where the images are sourced from. Used for naming output
3. CSV_PATH - path to where the metadata csv will be
4. IMAGE_DIR - directory to save images to. There will be up to 3 folders
    - healthy 
    - bleached
    - dead
5. METADATA_JSON - parsed annotation metadata json file (obtained from ```parse_metadata.py```)
    - Should be in the `./outputs/metadata` folder. General path structure: `./outputs/metadata/<source_name>.json`
6. HREFS_JSON - json file of all image links corresponding to a specific source (obtained from ```scrape_image_refs.py```)
    - Should be in the outputs/image_links folder. General path structure: `./outputs/image_links/<source_name>_image_links.json`

This script is used to use the image links parsed from ```scrape_image_refs.py``` and save the full image to a folder corresponding to its label. 
This script also saves a metadata csv for each coralnet source in it's image directory. This csv has info on the image names, dates, locations and labels. 

Params 2, 5, and 6 should be changed for each CoralNet source. The image dir name (param 4) will be based on the coralnet source name given by param 2. 
Params 1 and 3 should not need to be changed. 
