import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

data=pd.read_csv('PriceCategorized.csv',header=0)



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

features=list(zip(host_is_superhost,neighbourhood_cleansed,room_type,accommodates,bathrooms_text,bedrooms,
beds,minimum,maksimum,review_scores_accuracy,review_scores_cleanliness,review_scores_checkin,
review_scores_communication,review_scores_location,review_scores_value,instant_bookable,
calculated_host_listings_count))
target=data.price
X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2)


model= KNeighborsClassifier()
model.fit(X_train,y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='euclidean',
metric_params=None, n_jobs=1, n_neighbors=50, p=2, weights= 'uniform')


expected=target 
predicted=model.predict(features)
print(metrics.classification_report(expected,predicted))
