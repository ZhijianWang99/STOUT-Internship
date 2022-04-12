# Import modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data frame
df = pd.read_csv("casestudy.csv")
df = df.drop(df.columns[0], axis=1)

years = sorted(list(set(df["year"])))


total_revenue_list = list()
for y in years:
    df_y = df[df["year"] == y]
    total_revenue = sum(df_y["net_revenue"])
    total_revenue_list.append(total_revenue)
    print("Total Revenue in %d: %.2f" % (y, total_revenue))
temp_df = df.groupby("year").sum().reset_index()
temp_df.to_csv("total_revenue.csv")

plt.figure(figsize=(13, 8))
new_revenue_list = list()
for y in years:
    if y == years[0]:
        continue
    df_y = df[df["year"] == y]
    df_previous_y = df[df["year"] == y - 1]
    new_customs = set(df_y["customer_email"]) - set(df_previous_y["customer_email"])
    new_revenue = df_y[df_y["customer_email"].isin(new_customs)]
    print("New Custom Revenue in %d" % (y))
    sns.kdeplot(new_revenue["net_revenue"])

    display(new_revenue)
    new_revenue.to_csv("New Custom Revenue in %d.csv" % (y))
plt.legend(["net_revenue for new customers in 2016", "net_revenue for new customers in 2017"])

plt.show()



plt.figure(figsize = (13,8))
for y in years:
    if y == years[0]:
        continue
    df_y = df[df["year"] == y]
    df_previous_y = df[df["year"] == y-1]
    existing_customs = set(df_y["customer_email"]).intersection(set(df_previous_y["customer_email"]))
    current_year_df = df_y[df_y["customer_email"].isin(existing_customs)][["customer_email","net_revenue"]]
    last_year_df = df_previous_y[df_previous_y["customer_email"].isin(existing_customs)][["customer_email","net_revenue"]]
    merged_table = pd.merge(current_year_df, last_year_df, on = "customer_email")
    merged_table['revenue_growth'] = merged_table["net_revenue_x"] - merged_table["net_revenue_y"]
    merged_table = merged_table[["customer_email", 'revenue_growth']]
    print("Revenue Growth in %d" % (y))
    display(merged_table)
    merged_table.to_csv("Revenue Growth in %d.csv" % (y))
    sns.kdeplot(merged_table['revenue_growth'])
plt.legend(["Revenue Growth in 2016", "Revenue Growth in 2017"])

plt.show()

# Existing Customer Revenue Current Year
# Existing Customer Revenue Prior Year
for y in years:
    if y == years[0]:
        continue
    df_y = df[df["year"] == y]
    df_previous_y = df[df["year"] == y - 1]
    existing_customs = set(df_y["customer_email"]).intersection(set(df_previous_y["customer_email"]))
    current_year_df = df_y[df_y["customer_email"].isin(existing_customs)][["customer_email", "net_revenue"]]
    last_year_df = df_previous_y[df_previous_y["customer_email"].isin(existing_customs)][
        ["customer_email", "net_revenue"]]
    merged_table = pd.merge(current_year_df, last_year_df, on="customer_email")
    merged_table.rename(columns={'net_revenue_x': 'current_year', 'net_revenue_y': 'last_year'}, inplace=True)
    print("Existing Customer Revenue in %d" % (y))
    merged_table.to_csv("Existing Customer Revenue in %d.csv" % (y))
    display(merged_table)

for y in years:
    if y == years[0]:
        continue
    df_y = df[df["year"] == y]
    df_previous_y = df[df["year"] == y-1]
    print("Total Customer Current in year %d: %d" %(y, len(df_y)))
    print("Total Customer Prior in year %d: %d" %(y, len(df_previous_y)))


# New Customers
# Lost Customers
for y in years:
    if y == years[0]:
        continue
    df_y = df[df["year"] == y]
    df_previous_y = df[df["year"] == y-1]
    new_customers = set(df_y["customer_email"]) - set(df_previous_y["customer_email"])
    lost_customers = set(df_previous_y["customer_email"]) - set(df_y["customer_email"])
    new_df = df_y[df_y["customer_email"].isin(new_customers)][["customer_email","net_revenue"]]
    lost_df = df_previous_y[df_previous_y["customer_email"].isin(lost_customers)][["customer_email","net_revenue"]]
    print("New Customers in %d" % (y))
    new_df.to_csv("New Customers in %d.csv" % (y))
    display(new_df)
    print("Lost Customers in %d" % (y))
    lost_df.to_csv("Lost Customers in %d.csv" % (y))
    display(lost_df)