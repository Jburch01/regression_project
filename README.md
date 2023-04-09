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

<table style="width:100%">
  <tr>
    <th>Feature</th>
    <th>Values</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>tax_value</td>
    <td>integer</td>
    <td>The cost of the property</td>
  </tr>
  <tr>
    <td>Bedrooms</td>
    <td>integer</td>
    <td>The amount of bedrooms the property has</td>
  </tr>
  <tr>
    <td>Bathrooms</td>
    <td>integer</td>
    <td>The amount of bathrooms the property has</td>
  </tr>
  <tr>
    <td>Area</td>
    <td>integer</td>
    <td>The area (squarefeet) the property has</td>
  </tr>
  <tr>
    <td>Fips</td>
    <td>integer</td>
    <td>The county and state code</td>
  </tr>
  <tr>
    <td>Sale_date</td>
    <td>date yyyy/mm/dd str</td>
    <td>The sale date of the property</td>
  </tr>
</table>

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
