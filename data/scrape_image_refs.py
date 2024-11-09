from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import UnexpectedAlertPresentException, NoAlertPresentException
from selenium.webdriver.support import expected_conditions as EC
import time
import json

'''
Script to use selenium and get all the image links within a source's images page. 
General format of the url: https://coralnet.ucsd.edu/source/<source id>/browse/images/


SCRIPT PARAMS - 
    DRIVER_PATH - path to the chrome driver used for selenium. Make sure the driver version and chrome version match
    PAGE_URL - url to the images page for a given coralnet source. The page will be populated with thumbnails of the images in that source (20 per page)
    SOURCE_NAME - name of the source (used for output filename)
'''
# no need to change this 
DRIVER_PATH = './chromedriver-win64/chromedriver.exe'

# change these
PAGE_URL = 'https://coralnet.ucsd.edu/source/4430/browse/images/'
SOURCE_NAME = "National Park of American Samoa"

def handle_alert(driver):
    """Check for an alert and dismiss it if present."""
    try:
        alert = driver.switch_to.alert
        print(f"Alert detected and dismissed: {alert.text}")
        alert.dismiss()
    except NoAlertPresentException:
        pass  # No alert to dismiss, so continue normally


if __name__ == "__main__":

    service = Service(DRIVER_PATH)
    driver = webdriver.Chrome(service=service)
    driver.get(PAGE_URL)
    
    wait = WebDriverWait(driver, 30)

    # selector for the span that wraps the image thumbnails
    thumb_wrapper_selector = 'span.thumb_wrapper'
    # locator for the next page input element
    next_button_selector = 'input.page[value=">"]'

    # list of image hrefs
    all_image_links = []
    i = 1
    try:
        while True:
            # handle possible alert popups from coralnet
            handle_alert(driver)


            # Scrape the images on the current page
            thumb_wrappers = driver.find_elements(By.CSS_SELECTOR, thumb_wrapper_selector)
            for thumb_wrapper in thumb_wrappers:
                # Locate the <a> tag within each thumb wrapper and get its href attribute
                link = thumb_wrapper.find_element(By.TAG_NAME, 'a')
                image_href = link.get_attribute('href')
                print(f"Found href {i}: {image_href}")
                all_image_links.append(image_href)
                i += 1

            # check if the next button exists
            next_button = driver.find_elements(By.CSS_SELECTOR, next_button_selector)
            if next_button:
                # if the next button exists, find and submit the form containing it
                next_button[0].submit()
                time.sleep(10)  # wait for content to load

                # handle alert that may appear when clicking next
                handle_alert(driver)
            else:
                print("No more pages to load.")
                break
    finally:
        driver.quit()

    print(f"Total images scraped: {len(all_image_links)}")
    # save image hrefs for source to json
    with open(f'./outputs/image_links/{SOURCE_NAME}_image_links.json', 'w') as f:
        json.dump(all_image_links, f, indent=4)