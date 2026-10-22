# Cook County Housing Price Prediction

**Course:** Data 100, Principles and Techniques of Data Science (UC Berkeley, Summer 2024), Projects A1 and A2

**Question:** Can a linear model predict home sale prices well enough for property-tax assessment, and is "accurate" the same as "fair"?

## Data
About 205,000 residential sales from the **Cook County Assessor's Office (CCAO)** in Illinois, with 62 columns covering lot and building size, construction details, location, assessor estimates and sale price.

## What I did
**Part 1: exploration and feature engineering**
- Explored the heavily right-skewed sale price distribution and modeled **log sale price** instead.
- Removed outliers, extracted the number of bedrooms from free-text property descriptions with string parsing, flagged the most expensive neighborhoods (by median log price) as a binary feature, and one-hot encoded categorical columns such as roof material.
- Checked which features had the strongest linear relationship with log price.

**Part 2: building and evaluating my own model**
- Built a reusable feature-engineering pipeline function that applies the same steps to the training and test sets and only filters rows on the training side.
- Chose features by plotting each candidate against log price, log-transforming the size and estimate columns that looked linear in log space, and one-hot encoding basement and apartment indicators.
- Compared feature sets with **5-fold cross-validation** before fitting a final **ordinary least squares** model.
- Evaluated errors by price segment and discussed what positive and negative residuals mean for individual homeowners' property taxes.

## Results
- Training RMSE was about **$183k**, and 5-fold cross-validation RMSE was about **$182k**.
- Errors were much larger for homes above the median price (RMSE about $237k) than below it (about $70k). A single average error can hide who the model serves poorly.

## What I learned
- Log transforms and careful outlier handling matter more than adding many features.
- A low overall RMSE doesn't make an assessment model fair. Errors that differ by price range or neighborhood shift the tax burden between groups.

## Skills and tools
Python, pandas, NumPy, scikit-learn (LinearRegression, KFold, cross_val_score, OneHotEncoder), Matplotlib, Seaborn, EDA, log transforms, outlier removal, feature engineering pipelines, cross-validation, RMSE, model fairness.
