from etsy import Etsy
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

api = Etsy(key_file='./.env')

shops = []

for x in range(0,2):
    shops.append(api.findAllShops(shop_name='photography', limit=1))

shops

shops_data = pd.DataFrame.from_records(shops)
shops_data = shops_data.set_index('shop_name')

shops_data.info()

shops_data.head(3)


// This is what we are trying to learn
shops_data["num_favorers"]

# In[ ]
shops_data.plot(kind='scatter', x='listing_active_count', y='num_favorers', figsize=(5,3))
plt.show()

shops_data.hist(bins=50, figsize=(20,10))
plt.show()




# split training set
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(shops_data, test_size=0.2, random_state=42)
print(len(train_set), len(test_set))






from pandas.tools.plotting import scatter_matrix

attributes = ["listing_active_count", "last_updated_tsz", "creation_tsz", "digital_listing_count"]
scatter_matrix(train_set[attributes], figsize=(15, 8))
plt.show()
