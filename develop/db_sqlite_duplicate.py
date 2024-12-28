import os
import shutil
import sqlite3

import pandas as pd
import requests

db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "/mnt/c/Users/okapr/OneDrive/Documents/db-sqlite/travel2.sqlite"
# The backup lets us restart for each tutorial section
backup_file = "/mnt/c/Users/okapr/OneDrive/Documents/db-sqlite/travel2.backup.sqlite"
overwrite = False

if overwrite or not os.path.exists(local_file):
    response = requests.get(db_url)
    response.raise_for_status()  # Ensure the request was successful
    with open(local_file, "wb") as f:
        f.write(response.content)
    # Backup - we will use this to "reset" our DB in each section
    shutil.copy(local_file, backup_file)

# Convert the flights to present time for our tutorial
def update_dates(file):
    shutil.copy(backup_file, file)
    dbconn = sqlite3.connect(file)

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", dbconn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", dbconn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
        + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, dbconn, if_exists="replace", index=False)

    del df
    del tdf

    dbconn.commit()
    dbconn.close()

    return file

update_dates(local_file)
