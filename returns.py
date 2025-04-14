import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Define symbols
symbols = ['^GSPC', '^IXIC', 'GLD', 'AAPL', 'MSFT', 'AMZN', 'JPM', 'WMT', 'UNH', 'V', 
           'PG', 'JNJ', 'HD', 'MRK', 'CVX', 'KO', 'CRM', 'CSCO', 'MCD', 
           'AMGN', 'IBM', 'AXP', 'CAT', 'VZ', 'DIS', 'GS', 'INTC', 'HON', 
           'BA', 'NKE', 'MMM', 'TRV', 'DOW']

# Date range
start = (datetime.now() - relativedelta(months=12)).strftime('%Y-%m-%d')
end = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

# Fetch data and calculate returnsh
data = yf.download(symbols, start=start, end=end)['Close']
returns = data.pct_change().dropna()

print(returns)

###

# Train and predict probability of stocks going up tomorrow.

predictions = {}

for stock, df in features_all.items():
    X = df.drop(columns='target')
    y = df['target']
    
    # Train/test split
    X_train, X_test = X[:-1], X[-1:]
    y_train = y[:-1]

    # Model: Random Forest (or swap for XGBClassifier)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict probability for the most recent day
    prob_up = model.predict_proba(X_test)[0][1]  # P(up tomorrow)
    predictions[stock] = prob_up


# Display Results (Sorted)

probs_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['P_Up_Tomorrow'])
probs_df['P_Down_Tomorrow'] = 1 - probs_df['P_Up_Tomorrow']
probs_df = probs_df.sort_values(by='P_Up_Tomorrow', ascending=False)

print(probs_df)  # Top 10 most likely to rise tomorrow

###

import plotly.express as px

# Reset index so 'stock' becomes a column
probs_df = probs_df.reset_index().rename(columns={'index': 'Stock'})

# Create horizontal bar chart
fig = px.bar(
    probs_df,
    x='P_Up_Tomorrow',
    y='Stock',
    orientation='h',
    title='Predicted Probability of Stock Going Up Tomorrow',
    labels={'P_Up_Tomorrow': 'Probability (↑ Tomorrow)', 'Stock': 'Ticker'},
    color='P_Up_Tomorrow',
    color_continuous_scale='Greens'
)

fig.update_layout(
    yaxis=dict(autorange='reversed'),  # Highest probability at the top
    template='plotly_white',
    xaxis_tickformat='.0%',
    height=800
)

fig.show()

# End!



