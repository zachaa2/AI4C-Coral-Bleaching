import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":
    # URL of the page containing the table
    url = "https://coralnet.ucsd.edu/label/list/"

    # Send a GET request to fetch the page content
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table with id "label-table"
    table = soup.find('table', id='label-table')

    # Initialize lists for different coral types
    healthy_list = []
    dead_list = []
    bleaching_list = []

    # Iterate over each row in the table body
    for row in table.find_all('tr'):
        # Get all 'td' tags in the row
        tds = row.find_all('td')

        if len(tds) < 5:
            continue  # Skip rows with insufficient columns

        # Extract the label name from the first 'td' tag
        label_name = tds[0].find('a').get_text(strip=True)

        # Extract the functional group from the second 'td' tag
        functional_group = tds[1].get_text(strip=True)

        # Only care about "Hard coral" functional group
        if functional_group == "Hard coral":
            # Get the short code from the fifth 'td' tag
            short_code = tds[4].get_text(strip=True)

            # add labels to correct list
            if "bleach" in label_name.lower():
                bleaching_list.append(short_code)
            elif "dead" in label_name.lower():
                dead_list.append(short_code)
            else:
                healthy_list.append(short_code)

    # Write the short codes to respective text files
    with open("./labels/healthy_coral_codes.txt", "w") as healthy_file:
        healthy_file.write("\n".join(healthy_list))

    with open("./labels/dead_coral_codes.txt", "w") as dead_file:
        dead_file.write("\n".join(dead_list))

    with open("./labels/bleaching_coral_codes.txt", "w") as bleaching_file:
        bleaching_file.write("\n".join(bleaching_list))

    print("Successfully saved short code for Hard coral labels.")