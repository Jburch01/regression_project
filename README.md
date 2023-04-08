# Zillow 

---

# Project Description



To build a regression model to predict the property tax assessed values ('taxvaluedollarcnt') of Single Family Properties that had a transaction during 2017.


# Project Goal
--- 
- Discover some potential drivers for property assessed values
- Delvelope ML regresson  models for property assessed values prediction
- Reccomend some key factors to predict property assessed values



# Initial Thoughts
---
My inital thoughts is area of homes and number of bedrooms/bathroom will be a driver for 
property assessed values



# Planning
---
- ### Acuire data 
- ### Prep/clean the data
    - Remove outliers 
    - Split data into train, validate, and test
- ### Explore the data
    - Discover potentil drivers 
    - Create hypothesis driver correlation
    - Preform Statistical Test on drivers
- ### Create Models for property assessed value prediction
    - Use models on train and validate data
    - Measure Models effectiveness on train and validate
    - Select best performing model for test
- ### Draw Conclusions 


# Data Dictionary 

| Feature | Values | Description                                  |
|-------------|:-----------:|----------------------------------------------|
| tax_value    | integer       | The cost of the property       |
| Bedrooms |    integer    | The amount of bedrooms the property has |
| Bathrooms | integer   | The amount of bathrooms the property has  |
| Area| integer  | The area (squarefeet) the property has |
| Fips| integer | The county and state code |
| Sale_date | date yyyy/mm/dd str | The sale date of the property |
# Steps to Reproduce 
- Clone repo
- Accqire data from SQL data base (must have credentials!)
- Use env file template (instructions inside)
- Run notebook

# Takeaways and Conclusions
- Area and bathrooms have a higher corelation with tax_value
- Bedrooms, bathrooms and area(squarefeet) are not enough feature to accurately predict home tax_value

# Recommendations 
- Acquire more data on homes as potential features 
- Build models using specific location (fips) seperately 
