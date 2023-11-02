from src.logger import logging
from src.exception import CustomException

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import json


class DbConnect:
    def __init__(self):
        logging.info("DBConnect initialized")

        # This secure connect bundle is autogenerated when you download your SCB,
        # if yours is different update the file name below
        cloud_config = {
            'secure_connect_bundle': './notebooks/db/secure-connect-db-census-income.zip'
        }

        # This token JSON file is autogenerated when you download your token,
        # if yours is different update the file name below
        with open("./notebooks/db/db_census_income-token.json") as f:
            secrets = json.load(f)

        CLIENT_ID = secrets["clientId"]
        CLIENT_SECRET = secrets["secret"]

        logging.info(CLIENT_ID)
        auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
        self.cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        self.session = self.cluster.connect('keyspace_census')

        row = self.session.execute(
            "select release_version from system.local").one()
        if row:
            print(row[0])
            logging.info("DBConnect sucessfully.....")
        else:
            print("An error occurred.")
            logging.info("DBConnect found an error...")

    def load_data(self):
        query = "SELECT * FROM income_cleaned;"

        results = self.session.execute(query)

        # for row in results:
        #     print(row)
        # Close the Session object.
        self.session.shutdown()

        # Close the Cluster object.
        self.cluster.shutdown()

        return results


# if __name__ == "__main__":
#     obj = DbConnect()
#     results = obj.load_data()
#     for row in results:
#         print(row)
