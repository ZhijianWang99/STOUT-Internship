# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import helpers
from tensorflow import feature_column
from tensorflow.keras import layers
import plotly.express as px


df = pd.read_csv("loans_full_schema.csv")
temp_df = df.groupby(['state']).mean().reset_index()


# reference: https://towardsdatascience.com/simplest-way-of-creating-a-choropleth-map-by-u-s-states-in-python-f359ada7735e

fig = px.choropleth(temp_df,
                    locations='state',
                    locationmode="USA-states",
                    color='annual_income',
                    color_continuous_scale="Viridis_r",
                    scope = "usa"
                    )
fig.show()

# reference: https://towardsdatascience.com/simplest-way-of-creating-a-choropleth-map-by-u-s-states-in-python-f359ada7735e

fig = px.choropleth(temp_df,
                    locations='state',
                    locationmode="USA-states",
                    color='interest_rate',
                    color_continuous_scale="Viridis_r",
                    scope = "usa"
                    )
fig.show()

temp = df.replace(to_replace='None', value=np.nan).dropna()
category_list = list()
numeric_list = list()
for n in temp.columns:
    if type(temp[n][37]) != str:
        numeric_list.append(n)
    else:
        category_list.append(n)
category_list.remove("emp_title")
category_list.remove("state")
category_list.remove("verified_income")
category_list.remove("sub_grade")
for n in category_list:
    plt.figure(figsize=(13, 8))
    if n == "grade":
        ax = seaborn.violinplot(x=n,
                                y="interest_rate",
                                data=df, order=["A", "B", "C", "D", "E", "F"])
    else:
        ax = seaborn.violinplot(x=n,
                                y="interest_rate",
                                data=df)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)


plt.figure(figsize = (13,8))
seaborn.distplot(df["interest_rate"], hist=True, kde=True,
             bins=int(180/5), color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.show()


plt.figure(figsize = (10,10))
temp_df = df.groupby("grade").count().reset_index()
colors = seaborn.color_palette('pastel')[0:len(df["grade"].unique())]
labels = sorted(df["grade"].unique())
explode = [0,0,0,0,0,0.2, 0.4]
patches=plt.pie(temp_df["emp_title"], labels = labels, colors = colors, autopct='%1.1f%%', explode = explode)
plt.title("Grade percentage")
plt.show()

plt.figure(figsize = (10,10))
temp_df = df.groupby("loan_purpose").count().reset_index()
colors = seaborn.color_palette('pastel')[0:len(df["loan_purpose"].unique())]
labels = sorted(temp_df["loan_purpose"].unique())
explode = (0.2,0,0,0,0,0,0,0.2,0,0.5,0.2,0)
plt.pie(temp_df["emp_title"], labels = labels, colors = colors, autopct='%1.1f%%', explode = explode)
plt.title('loan_purpose_percentage')
plt.show()

plt.figure(figsize = (13,8))
numeric_df = df[numeric_list]
seaborn.heatmap(numeric_df.corr());
plt.show()

#https://stackoverflow.com/questions/46433588/pandas-drop-rows-columns-if-more-than-half-are-nan
def delCol(df1):
    df1 = df1.dropna(axis=1, thresh=df.shape[0]//2)
    return df1


predict_df = df
predict_df["verified_income"] = predict_df["verified_income"].replace(["Source Verified"], 'Verified')
predict_df["verification_income_joint"] = predict_df["verification_income_joint"].replace(["Source Verified"],
                                                                                          'Verified')
useless_column = ["num_accounts_30d_past_due", "num_accounts_120d_past_due", "emp_title", "issue_month", "sub_grade"]
predict_df = predict_df.drop(useless_column, axis=1)
predict_df.dropna(axis=1, thresh=df.shape[0] // 2, inplace=True)
tempName = predict_df.columns.values.tolist()
for name in tempName:
    if type(predict_df.at[1, name]) != str:
        myValue = int(predict_df[name].mean())
        predict_df[name] = predict_df[name].fillna(int(myValue))
    else:
        predict_df[name] = predict_df[name].fillna(predict_df[name].mode())
train, test = train_test_split(predict_df, test_size=0.2)

y_train, x_train = train["interest_rate"], train.drop("interest_rate", axis = 1)
y_train = y_train.values.reshape((-1,1))
y_test, x_test = test["interest_rate"], test.drop("interest_rate", axis = 1)
y_test =  y_test.values.reshape((-1,1))

#https://www.tensorflow.org/tutorials/structured_data/feature_columns
def df_to_dataset(df, shuffle=True, batch_size=32):
    df = df.copy()
    result = df.pop('interest_rate')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), result))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
print(len(train), 'train examples')
print(len(test), 'validation examples')

for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of targets:', label_batch )

# numerical_columns
feature_columns = list()
for header in ['annual_income', 'debt_to_income',
               "delinq_2y", 'inquiries_last_12m',
               'total_credit_lines', 'open_credit_lines',
               'total_credit_limit', 'total_credit_utilized',
               'num_collections_last_12m', 'num_historical_failed_to_pay',
               'current_accounts_delinq', "total_collection_amount_ever",
               "current_installment_accounts", "accounts_opened_24m",
               "months_since_last_credit_inquiry",'num_satisfactory_accounts',
               'num_active_debit_accounts', 'total_debit_limit',
               'num_total_cc_accounts', 'num_open_cc_accounts',
               'num_cc_carrying_balance', 'num_mort_accounts',
               'account_never_delinq_percent', 'tax_liens',
               "public_record_bankrupt", 'loan_amount',
               "installment", 'balance',
               "paid_total", 'paid_principal',
               'paid_interest', "paid_late_fees"]:
    feature_columns.append(feature_column.numeric_column(header))


# indicator_columns
for col_name in ["homeownership", 'verified_income', 'loan_purpose', 'application_type',
               "grade", 'loan_status', "initial_listing_status", "disbursement_method", "state"]:
    categorical_column = feature_column.categorical_column_with_vocabulary_list(col_name, predict_df[col_name].unique())
    indicator_column = feature_column.indicator_column(categorical_column)
    feature_columns.append(indicator_column)


# bucket categories
emp_length = feature_column.numeric_column("emp_length")
emp_buckets = feature_column.bucketized_column(emp_length, boundaries=[2.5, 4.5, 6.5, 8.5])
feature_columns.append(emp_buckets)
earliest = feature_column.numeric_column("earliest_credit_line")
earliest_bucket = feature_column.bucketized_column(earliest, boundaries=[1970, 1980, 1990, 2000, 2010])
feature_columns.append(earliest_bucket)
term = feature_column.numeric_column("term")
term_bucket = feature_column.bucketized_column(term, boundaries=[37])
feature_columns.append(term_bucket)
len(feature_columns)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128),
    layers.Dense(128),
    layers.Dense(64),
    layers.Dense(64),
    layers.Dense(32),
    layers.Dense(32),
    layers.Dense(16),
    layers.Dense(16),
    layers.Dense(8),
    layers.Dense(8),
    layers.Dense(4),
    layers.Dense(4),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(train_ds, validation_data = test_ds, epochs = 300)


real_value = test["interest_rate"]
predicted_value = model.predict(test_ds)
predicted_value = predicted_value.reshape((1,-1))
predicted_value = predicted_value[0]
predicted_value


show_df = real_value.to_frame()
show_df["predicted_value"] = predicted_value
show_df

fig = seaborn.kdeplot(show_df["predicted_value"], shade=True, color="r")
fig = seaborn.kdeplot(show_df["interest_rate"], shade=True, color="b")
plt.legend(["predicted_value", "interest_rate"])
plt.show()