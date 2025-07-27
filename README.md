CARDEKHO Used Car Price Prediction
The main objective of this project is to build a ML model to accurately predict the used car prices by analyzing the features like model,fuel type, age of car, kms_driven, mileage, engine displacement etc.

Data Preprocessing: Extensive cleaning and feature engineering on car data, including encoding categorical variables. Exploratory Data Analysis (EDA): Visualizations using Matplotlib and Seaborn were used to further understand the data and some insights were derived. Machine Learning Models:Extra Tree Regressor, Linear Regression, Decision Tree, Random Forest, and XGBRegressor were evaluated,Comparitively the xgboost (XGBRegressor) Model gave a better R2 score. Hence after HyperParameter tuning the model using the RandomizedSearchCV , the model's r2 score increased and the MSE decreased to some extent. The model was then saved as a pickle file and was deployed using Streamlit Apllication to predict the Prices.. Streamlit App: A user-friendly interface where users can input car details to get instant price predictions. Integrated with a hyper tuned xgbRegressor model for high accuracy.

Python Libraries: Pandas,numpy,Scikit-learn,Streamlit.
