import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

dataSet = pd.read_csv("Social_Network_Ads.csv")

x = dataSet.iloc[:10,0:10]
y = dataSet.iloc[:10, :10]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

st_x = StandardScaler()

x_train = st_x.fit_transform(x_train)
x_train = st_x.fit_transform(x_test)