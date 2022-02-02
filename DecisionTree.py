import pandas as pd
from pandas.core.algorithms import mode
from scipy.sparse.construct import random
from sklearn import preprocessing
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



data=pd.read_csv('PriceCategorized.csv')
from sklearn.model_selection import train_test_split
le = preprocessing.LabelEncoder()

host_is_superhost=data['host_is_superhost']
neighbourhood_cleansed=data['neighbourhood_cleansed']
room_type=data['room_type']
price=data['price']
accommodates=le.fit_transform(data['accommodates'])
bathrooms_text=le.fit_transform(data['bathrooms_text'])                                
bedrooms=le.fit_transform(data['bedrooms'])
beds=le.fit_transform(data['beds'])
minimum=le.fit_transform(data['minimum_nights'])
maksimum=le.fit_transform(data['maximum_nights'])
review_scores_accuracy=le.fit_transform(data['review_scores_accuracy'])
review_scores_cleanliness=le.fit_transform(data['review_scores_cleanliness'])
review_scores_checkin=le.fit_transform(data['review_scores_checkin'])
review_scores_communication=le.fit_transform(data['review_scores_communication'])
review_scores_location=le.fit_transform(data['review_scores_location'])
review_scores_value=le.fit_transform(data['review_scores_value'])
instant_bookable=data['instant_bookable']
calculated_host_listings_count=le.fit_transform(data['calculated_host_listings_count'])

features=list(zip(host_is_superhost,neighbourhood_cleansed,room_type,accommodates,bathrooms_text,bedrooms,beds,minimum,maksimum,review_scores_accuracy,review_scores_cleanliness,review_scores_checkin,review_scores_communication,review_scores_location,review_scores_value,instant_bookable,calculated_host_listings_count))
target=data.price
X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2)

clf=tree.DecisionTreeClassifier()
clf.fit(features,target)
predictions=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, predictions))



