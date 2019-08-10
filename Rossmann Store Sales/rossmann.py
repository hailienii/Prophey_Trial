import pandas as pd
from fbprophet import Prophet

# importing data
df = pd.read_csv("train.csv",  low_memory=False)
test_df = pd.read_csv("test.csv", low_memory=False)

# remove closed stores and those with no sales
df = df[(df["Open"] != 0) & (df['Sales'] != 0)]

# sales for store number 1
sales = df[df.Store == 1].loc[:, ['Date', 'Sales']]

# reverse the order: from 2013 to 2015
sales = sales.sort_index(ascending=False)

# to datetime
sales['Date'] = pd.DatetimeIndex(sales['Date'])

# from the prophet documentation every variables should have specific names
sales = sales.rename(columns={'Date': 'ds', 'Sales': 'y'})

# create dataframe of holidays in the dataset
state_dates = df[(df.StateHoliday == 'a') | (df.StateHoliday == 'b')
                 & (df.StateHoliday == 'c')].loc[:, 'Date'].values
school_dates = df[df.SchoolHoliday == 1].loc[:, 'Date'].values

state = pd.DataFrame({'holiday': 'state_holiday',
                      'ds': pd.to_datetime(state_dates)})
school = pd.DataFrame({'holiday': 'school_holiday',
                       'ds': pd.to_datetime(school_dates)})

holidays = pd.concat((state, school))
holidays.head()

# set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet(interval_width=0.95, holidays=holidays,
                   daily_seasonality=True)
my_model.fit(sales)

# dataframe that extends into future 6 weeks
future_dates = my_model.make_future_dataframe(periods=6*7)


# predictions
forecast = my_model.predict(future_dates)

# preditions for last week
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)

# visualizing predicions
my_model.plot(forecast)
my_model.plot_components(forecast)

