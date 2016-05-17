#Expedia Hotel Analytics

##Advanced Big Data Analytics

Aditya Bagri         aab2234@columbia.edu                                            
Gautam Sihag         gs2835@columbia.edu   

                    

### Introduction

We will be working with [Expedia Hotel Recommendation Data sets](https://www.kaggle.com/c/expedia-hotel-recommendations/data). This data set has been provided by Expedia as logs of customer behavior. These include what customers searched for, how they interacted with search results (click/book), whether or not the search result was a travel package (hotel booking + flight ticket).
The data belongs to a Kaggle competition, and is a random selection from Expedia and is not representative of the overall statistics.
Kaggle competitions are a fantastic way to learn data science and to build portfolio. We personally use Kaggle to learn many data science concepts. We started out with Kaggle just for this class but we hope to win a competition soon.
Doing well in a Kaggle competition requires more than just knowing machine learning algorithms. It requires the right mindset, the willingness to learn, and a lot of data exploration. Many of these aspects aren’t typically emphasized in any tutorials.

####Motivation
Hotel industry is an industry where effective use of analytics can change dramatically how business is run. It is a data rich industry that captures huge volumes of data of different types, however, for most hoteliers data remains an underused and under-appreciated asset. And so, we used the Expedia data set hosted on Kaggle for a competition to explore and efficiently employs big data analytics to help generate actionable insights in the Hotel Industry. We found the motivation for this project at the prospect of bridging the gap between the hotel owners and the customers and assisting both parties to make the most of the resources available to them. The idea of pitching our results and analysis visualizations to a hotel company such as Expedia looking for tools for expanding the business based on data analytics is another incentive to take this up. The fact that these companies are making their data sets public means that they want data scientists to perform analytics based on hotel data and so we believe research in this field is highly valuable and potentially marketable.


### Data
The data we will be working with is publicly available and for competitions on kaggle at[Expeida Kaggle Datasets] (https://www.kaggle.com/c/expedia-hotel-recommendations/data).  



File descriptions:

- train.csv - the training set
- test.csv - the test set
- destinations.csv - hotel search latent attributes
- sample_submission.csv - a sample submission file in the correct format


The attributes of the train.csv file are:

- date_time	Timestamp	| string
- site_name	ID of the Expedia point of sale (i.e. Expedia.com, Expedia.co.uk, Expedia.co.jp, ...)	| int
- posa_continent	ID of continent associated with site_name	| int
- user_location_country	The ID of the country the customer is located	| int
- user_location_region	The ID of the region the customer is located	| int
- user_location_city	The ID of the city the customer is located	| int
- orig_destination_distance	Physical distance between a hotel and a customer at the time of search. A null means the distance could not be calculated	| double
- user_id	ID of user	| int
- is_mobile	1 when a user connected from a mobile device, 0 otherwise	| tinyint
- is_package	1 if the click/booking was generated as a part of a package (i.e. combined with a flight), 0 otherwise	| int
- channel	ID of a marketing channel	| int
- srch_ci	Checkin date	| string
- srch_co	Checkout date	| string
- srch_adults_cnt	The number of adults specified in the hotel room	| int
- srch_children_cnt	The number of (extra occupancy) children specified in the hotel room	| int
- srch_rm_cnt	The number of hotel rooms specified in the search	| int
- srch_destination_id	ID of the destination where the hotel search was performed	| int
- srch_destination_type_id	Type of destination	| int
- hotel_continent	Hotel continent	| int
- hotel_country	Hotel country	| int
- hotel_market	Hotel market	| int
- is_booking	1 if a booking, 0 if a click	| tinyint
- cnt	Numer of similar events in the context of the same user session	| bigint
- hotel_cluster	ID of a hotel cluster	| int

Attributes of Destination.csv file are:
- srch_destination_id	ID of the destination where the hotel search was performed	| int
- d1-d149	latent description of search regions	| double

This is data is clean and anonymized for user privacy, and does not include any sensitive information about the customers or hotels or country of search or country of travle.


#### Sources
1. [Expedia Datasets hosted on Kaggle](https://www.kaggle.com/c/expedia-hotel-recommendations/data)


### Tools & Languages

1. Jupyter
2. Matplotlib
3. Seaborn
4. pySpark
5. scikit-learn
6. Language Choice: Python
7. Environment: Docker Instance
8. Pandas
9. Numpy
 
### Results

1. To Explore the large data set for one feature at a time resulted in to create visualization to generate numerous insights and provided a mark of the Key Performance Indicators in the hospitality industry.
2. An analysis of adults and children numbers for the count of rooms booked gave an indication of trends such as typically two adults travel alone and also depicted how the number of rooms increase or decrease with the variance in the number adults/children.
3. Detailed inspection of the data resulted in us being able to define an interesting and unique attribute, search span, that we were able to correlate with several other attribute to generate informative insights for the hotel industry.


#### Contributing
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D 

## Acknowledgments

* We would like to use this platform to thank Professor Lin for such an informative and intellectually stimulating course, we definitely learned a great deal.
* Like to thank the TA’s for their constant guidance and assistance throughout the course of this project
* Also, we acknowledge Kaggle and Expedia for data set and numerous other Kagglers for their reviews and comments. 
