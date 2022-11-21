# zillow-project
Project: Single family residence price predictions. 
 
## Project Goal
Build the machine learning model that can predict single family residence prices based on the data from 2017.

## Steps to Reproduce
1) Clone this repo into your computer.
2) Acquire the data from databaase using your ```env.py``` file
3) Put the data in the file containing the cloned repo.
4) Run the ```zillow_project.ipynb``` file.

 
## Initial Thoughts
 
My initial hypothesis is that main price predictors going to be number of bathrooms and bedrooms.
 
## Project's pipeline
 

#### Aqcuire and prepare
1. Acquire the data from the ```zillow``` database. Transform the data to a Pandas dataframe to make it easy to use and manipulate in the Jupyter Notebook.
2. Prepare the data for exploration and analysis. Find out if there are some values missing and find a way to handle those missing values.
3. Change the data types if needed
4. Find if there are features that can be created to simplify the exploration process.
5. Handle the outliers.
6. Create a data dictionary.
7. Split the data into 3 data sets: train, validate and test data (56%, 24%, and 20% respectively)

#### Explore and pre-process
1. Explore the train data set through visualizations and statistical tests. 
2. Find which features have an impact on the house prices. 
2. Make the exploration summary and document the main takeaways.
3. Impute the missing values if needed.
4. Pick the features that can help to build a good prediction model.
5. Identify if new features have to be created.
6. Encode the categorical variables
7. Split the target variable from the data sets.
8. Scale the data prior to modeling.

#### Build a regression model
1. Pick the regression algorithms for creating the prediction model.
2. Create the models and evaluate regressors using the **RMSE** score on the train data set.
3. Pick five of the best performing models based on the RMSE score and evaluate them on the validation set.
4. Find out which model has the best performance: relatively high predicting power on the validation set and slight difference in the train and validation prediction results.
5. Make predictions for the test data set.
6. Evaluate the results.

*Drow conclusions*
 
## Data Dictionary


| Feature | Definition | Manipulations applied|Data Type|
|:--------|:-----------|:-----------|:-----------|
|<img width=50/>|<img width=200/>|<img width=50/>|<img width=100/>|
|||**Categorical Data**
|<img width=50/>|<img width=200/>|<img width=50/>|<img width=100/>|
|*county_name*| Names of the counties in the data set  | canged fips code into county names:'LA', 'Orange', 'Ventura'| category
|||**Numerical Data**
|<img width=50/>|<img width=100/>|<img width=50/>|<img width=150/>|
|*bedrooms*|  Number of bedrooms | Changed the type into integer| integer
|*bathrooms*|  Number of bathrooms | Half-bathrooms were turned into whole number, changed the type intoto integer| integer
|*sq_feet*| Squared feet of the house | Changed the type into integer| integer
|*lot_sqft*| Squared feet of the land | Changed the type into integer| integer
|*year_built*| Year the house was built | Changed the type into integer| integer
|*house_age*| Age of the house | Created the column by subtracting the year_built from 2017| integer
|*pools*| Number of pools | Replaced the null values with 0| integer
|**Target Data**
||<img width=150/>|<img width=550/>|
|**home_value** | **The Single Family Residence price** || **float**

#### Data preparation takeaways:
It was impossible to remove all outliers, it would decrease the data size dramatically. Two columns ```lot_sqft``` and ```home_value``` still contain lots of them. On top of this ```home_value``` contains some not realistic data(like home price being below $50K). This fact might negatively affect the model's performance.

#### Exploration Takeaways
- The mean price is more than $\$$80K higher that the median price
- The most common house prices are between $\$$50K and $100
- There is a significant difference in the house prices among counties. Houses in Orange county have the highest prices, while prices in Los Angeles are below the median.
- Houses *with* a pool are more expensive. Most of them have a price above the median.
- The most expensive houses without a pool are in Orange county and with a pool in Ventura county.
- There is a positive correlation between square footage and price.
- Ventura county have the strogest sq.footage / price relations.
- There is no correlation between the house age and its price in LA county while other counties have a strong negative correlation.

#### Modeling
- Gradient Boosting Regressor performed the best with the whole data set and with the Ventura county data. 
- Gradient Boosting Regressor is a good model in terms of prediction but doesn't return stable results. The RMSE scores vary a lot in all 3 sets.
- For stable results I would pick Random Forest Regressor or Lasso Lars Regressor.

### Conclusions and my thoughts how to improve the result
 - Overall my regression model performs good. Its predictions beat the baseline model by 23.5%
 - The model would perform even better if the data from LA county contained stronger relation between features and price. 
 - To imporove prediction results I would recommend to pull more features from the database and look for the ones that have a strong correlation with the price in LA county.