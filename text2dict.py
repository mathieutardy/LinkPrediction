import string
import re
import pickle
from nltk.corpus import stopwords
from stemming.porter2 import stem
import networkx as nx
from tqdm import tqdm


def ProcessAmazon(save_data_path):
    """
    This function opens the Amazond data as a text file.

    The loop allows us to store the relevant products metadata into a Python Dictionary,
    it will therefore be easier to generate graphical data with the obtained object.

    This takes approximately 5min.
    """

    fr = open("./data/amazon-meta.txt", "r", encoding="utf-8", errors="ignore")

    # Creating Dictionary to store data with relevant keys
    amazonProducts = {}
    (
        Id,
        ASIN,
        Title,
        Categories,
        Group,
        CoPurchased,
        SalesRank,
        TotalReviews,
        AvgRating,
        DegreeCentrality,
        ClustCoeff,
    ) = ("", "", "", "", "", "", 0, 0, 0.0, 0, 0.0)

    for line in tqdm(fr, position=0, leave=True):

        cell = line.strip()

        if cell.startswith("Id"):
            Id = cell[3:].strip()
        elif cell.startswith("ASIN"):
            ASIN = cell[5:].strip()
        elif cell.startswith("title"):
            Title = cell[6:].strip()
            Title = " ".join(Title.split())
        elif cell.startswith("group"):
            Group = cell[6:].strip()
        elif cell.startswith("salesrank"):
            SalesRank = cell[10:].strip()
        elif cell.startswith("similar"):
            temp = cell.split()
            CoPurchased = " ".join([el for el in temp[2:]])
        elif cell.startswith("categories"):
            temp = cell.split()
            Categories = " ".join(
                (fr.readline()).lower() for i in range(int(temp[1].strip()))
            )
            Categories = re.compile(
                "[%s]" % re.escape(string.digits + string.punctuation)
            ).sub(" ", Categories)
            Categories = " ".join(
                set(Categories.split()) - set(stopwords.words("english"))
            )
            Categories = " ".join(stem(word) for word in Categories.split())
        elif cell.startswith("reviews"):
            temp = cell.split()
            TotalReviews = temp[2].strip()
            AvgRating = temp[7].strip()

        elif cell == "":
            try:
                Metadata = {}
                if ASIN != "":
                    amazonProducts[ASIN] = Metadata
                Metadata["Id"] = Id
                Metadata["Title"] = Title
                Metadata["Categories"] = " ".join(set(Categories.split()))
                Metadata["Group"] = Group
                Metadata["CoPurchased"] = CoPurchased
                Metadata["SalesRank"] = int(SalesRank)
                Metadata["TotalReviews"] = int(TotalReviews)
                Metadata["AvgRating"] = float(AvgRating)
                Metadata["DegreeCentrality"] = DegreeCentrality
                Metadata["ClustCoeff"] = ClustCoeff
            except NameError:
                continue
            (
                Id,
                ASIN,
                Title,
                Categories,
                Group,
                CoPurchased,
                SalesRank,
                TotalReviews,
                AvgRating,
                DegreeCentrality,
                ClustCoeff,
            ) = ("", "", "", "", "", "", 0, 0, 0.0, 0, 0.0)

    fr.close()

    # Saving dictionary to pickle for re-use
    ftw = open(save_data_path, "wb")

    pickle.dump(amazonProducts, ftw)

    return amazonProducts
