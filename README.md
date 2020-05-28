
# Time Series Modeling:

In this project we are going to work with time series in order to predict the future property values for the top 5 zipcodes of San Jose Metro area in California based on the price of previous years.


# Project breakdown:

1. Importing necessary libraries and load the dataset.

2. Separating out the San Jose metro area from the rest of the dataset to work with.

3. Checking dataset for missing data, data types and placeholders.

4. Converting different zipcodes mean values into monthly ordered time-series from 1996-04-01 to 2018-04-01.

5. Checking time series stationarity.

6. Finding the best model parameters to work with.

7. Modeling and validating the results.

8. Predicting future property values using the model.



# Initial Data and zipcode based time series:

* Initial wide format dataset:

![alt text](https://github.com/FarnazG/project004/blob/master/images/df-head.png)


* Long format San Jose metro area dataset extracted from the initial dataset:

![alt text](https://github.com/FarnazG/project004/blob/master/images/san-jose-metro-df.png)



### The general trend of property values for different cities in San Jose metro area:

![alt text](https://github.com/FarnazG/project004/blob/master/images/general-trend-property-value.png)


* Zipcode based time series:

![alt text](https://github.com/FarnazG/project004/blob/master/images/zip-timeseries.png)



### Time series stationarity:

* Visualizing Autocorrelation and Partial-Autocollerlation plots of time series

![alt text](https://github.com/FarnazG/project004/blob/master/images/autocorrelation.png)

![alt text](https://github.com/FarnazG/project004/blob/master/images/partial_autocorrelation.png)



# The ARIMA model initial result:

* In-sample predictions for test data:

![alt text](https://github.com/FarnazG/project004/blob/master/images/in-sample-predictions.png)


* Out_of_sample predictions for future years, from 2018 to 2020: 

![alt text](https://github.com/FarnazG/project004/blob/master/images/out-of-sample-predictions.png)



### Model validation using residuals and density plots:

![alt text](https://github.com/FarnazG/project004/blob/master/images/model-validation.png)



# Interpreting Results:

### Finding best zipcodes within san jose metro area:

The prices arepredicted for the next 2 years of our data set, from 2018 to 2020.To find the 5 top most growing zipcodes:

Calculate the percentage of profit made from the investment within 4 years from 2016 to 2020:

profit = (Predicted property value on 2020 � investment value on 2016)/investment value on 2016

![alt text](https://github.com/FarnazG/dsc-mod-4-project-v2-1-online-ds-ft-120919/blob/master/images/Five_top_zipcode.png)

* The most profitable and highest return of investment goes to zipcode 95050, Santa Clara
* The second top profitable zipcode for investment is 94089, Sunnyvale
* Another 3 top ranked profitable zipcodes for investment are located in San Jose, 95131, 95130 and 95112.



## Non-technical Presentation

[time-series-modeling-presentation](https://github.com/FarnazG/project004/blob/master/time-series-modeling-presentation.pdf)



### Future Work and Recommendations:

1. Based on the data provided by zillow dataset, The generated ARIMA model is capable of predicting the average property value and its changes for any given period of time using the trends of previous years from 1996 to 2018.

2. For more accuracy, it seems better to keep short steps/periods of future predictions, longer periods of predictions will be more accurate after applying latest updates and changes to the dataset. 
