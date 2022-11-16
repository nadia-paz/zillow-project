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
1. Acquire the data from the ```zillow``` database. Transform the data to a Pandas data frame to make it easy to use and manipulate in the Jupyter Notebook.
2. Prepare the data for exploration and analysis. Find out if there are some values missing and find a way to handle those missing values.
3. Change data types if needed
4. Find if there are features that can be created to simplify the exploration process.
5. Handle the outliers.
6. Create a data dictionary.
7. Split the data into 3 data sets: train, validate and test data (56%, 24%, and 20% respectively)

#### Explore and pre-process
1. Explore the train data set through visualizations and statistical tests. 
2. Find which features that have an impact on the houses prices. 
2. Make the exploration summary and document the main takeaways.
3. Impute the missing values if needed.
4. Pick the features that can help to build a good predicting model.
5. Identify if new features have to be created.
6. Encode the categorical variables
7. Split the target variable from data sets.
8. Create scaled data sets for the modeling.

#### Build a regression model
1. Pick the regressor algorithms for creating the predicting model.
2. Create the models and evaluate regressors using the RMSE score on the train data sets.
3. Pick three best performing models based on the RMSE score and evaluate them on the validation set.
4. Find out which model has the best performance: relatively high predicting power on the validation test and slight difference in the train and validation prediction results.
5. Apply the predictions to the test data set.
6. Evaluate the results.

*Drow conclusions*
 
## Data Dictionary


| Feature | Definition |Manipulations applied|
|:--------|:-----------|:-----------|
|<img width=150/>|<img width=550/>|
|**Categorical Data**
||<img width=150/>|<img width=550/>|
|*county_name*| Names of the counties in the data set  | canged fips code into county names:'LA', 'Orange', 'Ventura'
|**Numerical Data**
||<img width=150/>|<img width=550/>|
|*bedrooms*|  Number of bedrooms | Changed the type into integer
|*bathrooms*|  Number of bathrooms | Half-bathrooms were turned into whole number, changed the type intoto integer
|*sq_feet*| Squared feet of the house | Changed the type into integer
|*lot_sqft*| Squared feet of the land | Changed the type into integer
|*year_built*| Year the house was built | Changed the type into integer
|*house_age*| Age of the house | Created the column by substracting the year_built from 2017
|*pools*| Number of pools | Replaced the null values with 0
|**Target Data**
||<img width=150/>|<img width=550/>|
|**home_value** | **The Single Family Residence price** | **float data type**