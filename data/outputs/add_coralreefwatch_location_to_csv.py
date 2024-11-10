import pandas as pd
import os
import requests

reef_watch_names_dic = {"Biscayne Bay": "southeast_florida", "Curacao":"abc_islands", "Pacific National Historical Park, Guam":"guam"}

def get_sst_90th_hs(row, dire):
    txt_file = open("images/{}/{}.txt".format(dire, row["CoralReefWatch location"]), "r")
    date = row["date"]

    lines = txt_file.readlines()

    # Step 3: Parse the file to find the data for the given date
    for line in lines:
        # Split the line into fields based on whitespace (assuming tab-delimited or space-separated)
        fields = line.split()

        if fields and (fields[0][:2]=="19" or fields[0][:2]=="20"):
            file_date = fields[0]+"-"+fields[1]+"-"+fields[2]

            # Step 4: Compare the date to the row's date
            if file_date == date:
                # Extract SST@90th_HS (Assumed to be in a known column, here we assume it's column 5)
                # Adjust the index according to the actual format of the file
                try:
                    sst_90th_hs = fields[5]  # Update index based on the actual column of SST@90th_HS
                    print(row["CoralReefWatch location"], date, sst_90th_hs)
                    return sst_90th_hs
                except IndexError:
                    print(f"Error: SST@90th_HS not found in the expected column for date {date}.")
                    return None

	# If the date is not found in the file
    print(f"Error: No data found for the date {date}.")
    return None

for dire in os.listdir("images"):
    csv_file_path = "images/{}/metadata.csv".format(dire)
    df = pd.read_csv(csv_file_path)
    df["CoralReefWatch location"] = reef_watch_names_dic[df.loc[0, "location"]]
    url = 'https://coralreefwatch.noaa.gov/product/vs/data/{}.txt'.format(df.loc[0, "CoralReefWatch location"])
    response = requests.get(url)
    with open("images/{}/{}.txt".format(dire, df.loc[0, "CoralReefWatch location"]), 'w') as f:
        f.write(response.text)
    df["SST@90th_HS"] = df.apply(get_sst_90th_hs, args=(dire, ), axis=1)
    df.to_csv(csv_file_path, index=False)
