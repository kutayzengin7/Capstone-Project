# Import Pandas, Scipy:
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
import io
import scipy.stats as sps
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Read CSV data file:
df_energy = pd.read_csv(r'C:\solar-energy-consumption.csv')

# Read CSV data file:
df_capacity = pd.read_csv(r'C:\installed-solar-pv-capacity.csv')

# Read CSV data file:
df_shr = pd.read_csv(r'C:\share-electricity-solar.csv')


# Fill the rows with missing values in the columns of interest:


def fill_missing_values(df, fill_values):
    return df.fillna(value=fill_values)


# Filtering out the rows in a Dataframe for column 'Year' < 2000:


def filter_by_year(df, year_column, threshold):
    return df[df[year_column] > threshold]

# Datasets include only countries, and dropping the rows has a missing value:


def clean_dataframe(df, entity_column, unwanted_entity):
    df_clean = df.dropna()
    return df_clean.drop(df_clean[df_clean[entity_column] == unwanted_entity].index)


values = {"Code": 'UN&RG'}

df_new = fill_missing_values(df_energy, values)
df_new = filter_by_year(df_new, 'Year', 1999)

# Filtering out the rows in the Dataframe and just leave the 'World' value:
df_wld = df_energy[['Entity', 'Code', 'Year', 'Electricity from solar (TWh)']].loc[df_energy['Entity'] == 'World']

# Dropping the rows that are less than 2000 for 'World' only:
df_wld = df_wld[['Entity', 'Code', 'Year', 'Electricity from solar (TWh)']].loc[df_wld['Year'] > 1999]

df_ctr = clean_dataframe(df_energy, 'Entity', 'World')

# Filtering out the rows in the Dataframe for column 'Year' < 2000:
df_ctr = df_ctr[['Entity', 'Code', 'Year', 'Electricity from solar (TWh)']].loc[df_ctr['Year'] > 1999]

# Filtering out the rows and only leave the 'World' value:
df_wld_capacity = df_capacity[['Entity', 'Code', 'Year', 'Solar energy capacity - GW']].loc[df_capacity['Entity'] ==
                                                                                            'World']

# Filtering out the rows and only leave the 'World' value:
df_shr = df_shr[['Entity', 'Code', 'Year', 'Solar - % electricity']].loc[df_shr['Entity'] == 'World']

# Dropping the rows in the Share Dataframe that are less than 2000 for 'World' only:
df_shr = df_shr[['Entity', 'Code', 'Year', 'Solar - % electricity']].loc[df_shr['Year'] > 1999]


# Choosing columns for skewness:


def select_column(df, column_name):
    return df[column_name]


year = select_column(df_new, 'Year')

Electricity = select_column(df_new, 'Electricity from solar (TWh)')

# Skewness calculation for 'Year' and 'Electricity from solar (TWh)' columns:


def calculate_skewness(df, columns):
    skewness_values = {}
    for column in columns:
        skewness_values[column] = sps.skew(df[column])
    return skewness_values


columns_to_analyze = ['Year', 'Electricity from solar (TWh)']

skewness_results = calculate_skewness(df_new, columns_to_analyze)


# Exporting to a text file
buffer = io.StringIO()
df_new.info(buf=buffer)
s = buffer.getvalue()
with open("df_new_info.txt", "w", encoding="utf-8") as f:
    f.write(s)
# Extracting the head of the Dataset:
summary = df_new.head()
summary.to_clipboard()

# Describe of the Dataset:
summary = df_new.describe()
summary.to_clipboard()

# Finding the column and row values:
print(df_new.shape)

# Creating histogram using Pandas Dataframe:
df_new.hist(column=["Electricity from solar (TWh)", "Year"])
# Show plot
plt.show()

# Drawing optimized box plot for a larger Dataframe.
box_plot = sns.catplot(
    data=df_new.sort_values("Year"),
    x="Year", y="Electricity from solar (TWh)", kind="boxen"
)
box_plot.set_xticklabels(rotation=90)
plt.show()

# Encoding categorical variable 'Entity' as a numeric value:
le = LabelEncoder()
df_new['Entity'] = le.fit_transform(df_new['Entity'])

# Drops the columns 'Code':
df_new1 = df_new.drop(['Code'], axis=1)
# Calculate the correlation matrix for the rest of the columns:
R = df_new1.corr()
# Obtain the squared correlation values:
df_new_corr = R ** 2
print(df_new_corr)

# Display the correlation heat map:
sns.set(color_codes=True)
sns.heatmap(df_new_corr, vmin=0, vmax=1, annot=True, fmt='f', cmap='coolwarm')
plt.show()

# Removing outliers from the Dataframe:
Q1 = df_new['Electricity from solar (TWh)'].quantile(0.25)
Q3 = df_new['Electricity from solar (TWh)'].quantile(0.75)
# Inter quartile range:
IQR = Q3 - Q1
# Find the minimum and maximum values:
Min = Q1 - 1.5 * IQR
Max = Q3 + 1.5 * IQR
# Remove outliers and return cleaned data in a new Dataframe:
df_new_remove = df_new.loc[(df_new['Electricity from solar (TWh)'] > Min) &
                           (df_new['Electricity from solar (TWh)'] < Max)]

# Perform visual linear regression analysis with seaborn:
sns.set(color_codes=True)
sns.regplot(x='Year', y='Electricity from solar (TWh)', data=df_new, label='regression', color='brown')
plt.legend()
plt.plot()
plt.show()

# Clear the figure and plot the residuals:
plt.clf()
sns.residplot(x='Year', y='Electricity from solar (TWh)', data=df_new, label='residuals')
plt.legend()
# Display the results:
plt.show()

# Clear the figure and plot the residuals:
plt.clf()
sns.residplot(x='Year', y='Electricity from solar (TWh)', data=df_ctr, label='residuals', color='purple')
plt.legend()
# Display the results:
plt.show()

# Use the stats module to perform linear regression:
slope, intercept, r_value, p_value, std_err = sps.linregress(df_new['Year'], df_new['Electricity from solar (TWh)'])
# Display the results:
print("slope= ", slope)
print("intercept= ", intercept)
print("r_value= ", r_value)
print("p_value= ", p_value)
print("std_err= ", std_err)

df_new['Electricity from solar (TWh)'] = df_new['Electricity from solar (TWh)'].astype(float).round(2)

# Defining training and test data:
train = df_new[df_new.Year < 2022]
test = df_new[df_new.Year >= 2022]

# Splitting the training and testing data:
x_year = df_new['Year']
x_year_train = train['Year']
x_year_test = test['Year']
y = df_new['Electricity from solar (TWh)']
y_train = train['Electricity from solar (TWh)']
y_test = test['Electricity from solar (TWh)']

# Order of the model:
order = 4

# Fitting the model:
model = np.poly1d(np.polyfit(x_year_train, y_train, order))

# Plotting the Polynomial model:
fig, ax1 = plt.subplots(figsize=(7, 4))
scat = ax1.scatter(x_year_train, y_train, marker='x', s=10, color='grey')
ax1.set_xlabel('Year')
ax1.set_ylabel('Electricity production')
ax1.tick_params('x', labelrotation=90)
ax1.grid(axis="y")
mod = ax1.plot(x_year_train, model(x_year_train), color='blue')
plt.title('Polynomial Model', color='blue')
plt.legend([scat, mod[0]], ['Training Data', 'Model'], loc='upper left')
plt.tight_layout()
plt.show()

# Linear Model:
# Define independent variable X as 'Year':
X = df_new['Year']
# Define dependent variable Y as 'Electricity from solar (TWh)':
Y = df_new['Electricity from solar (TWh)']
# Add the constant X, and store the result in Xc:
Xc = sm.add_constant(X)
# Fit the model:
model = sm.OLS(Y, Xc).fit()
# Define the testing data B:
B = [2025, 2028, 2030]
# Add constant to B ( and store the result in a new array Xtestc):
Xtestc = sm.add_constant(B)
# Make the prediction:
T = model.predict(Xtestc)
# Main program:
print(model.params)
print("Predictions:", list(zip(B, T)))
fig, ax = plt.subplots(figsize=(6, 4))
sm.graphics.plot_fit(model, 1, ax=ax)
plt.grid()
plt.show()

# Predicting energy productions for the 'World':
# Define independent variable X1 as 'Year':
X1 = df_wld['Year']
# Define dependent variable Y1 as 'Electricity from solar (TWh)':
Y1 = df_wld['Electricity from solar (TWh)']
# Add the constant A, and store the result in Xc1:
Xc1 = sm.add_constant(X1)
# Fit the model:
model = sm.OLS(Y1, Xc1).fit()
# Define the testing data B:
B1 = [2022, 2025, 2030]
# Add constant to B ( and store the result in a new array X1testc):
X1testc = sm.add_constant(B1)
# Make the prediction:
T1 = model.predict(X1testc)
# Main program:
print("Prediction:", list(T1))
print(model.params)
fig1, ax1 = plt.subplots(figsize=(6, 4))
sm.graphics.plot_fit(model, 1, ax=ax1)
plt.grid()
plt.show()

# Using the polynomial model for regression:
# Polynomial features:
degree = 2
X_poly = sm.add_constant(np.column_stack([df_wld['Year']] + [df_wld['Year'] ** i for i in range(2, degree + 1)]))
# Fit the polynomial model:
model1 = sm.OLS(df_wld['Electricity from solar (TWh)'], X_poly).fit()
# New data points for prediction
new_X = np.array([2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])
# Polynomial features for the new data:
new_X_poly = sm.add_constant(np.column_stack([new_X] + [new_X ** i for i in range(2, degree + 1)]))
# Make the prediction:
predictions = model1.predict(new_X_poly)
print(model1.params)
fig2, ax2 = plt.subplots(figsize=(6, 4))
sm.graphics.plot_fit(model1, 1, ax=ax2)
ax2.set_xlabel('Year')
plt.title('Electricity Generation From Solar', color='blue')
plt.grid()
plt.show()
print("Predictions:", list(zip(new_X, predictions)))

# Predicted electricity production values:
L = [1147, 1303, 1469, 1644, 1828, 2022, 2225, 2438, 2660]
n = len(L)
# Calculates the growth value for every year:
growth_rate = [((L[i] - L[i - 1]) * 100.0 / L[i - 1]) for i in range(1, n)]
print("Growth Rate:", growth_rate)

# Using the polynomial model for regression:
# Polynomial features:
degree = 2
X_poly = sm.add_constant(np.column_stack([df_wld_capacity['Year']] + [df_wld_capacity['Year'] ** i
                                                                      for i in range(2, degree + 1)]))
# Fit the polynomial model:
model1 = sm.OLS(df_wld_capacity['Solar energy capacity - GW'], X_poly).fit()
# New data points for prediction
new_X = np.array([2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])
# Polynomial features for the new data:
new_X_poly = sm.add_constant(np.column_stack([new_X] + [new_X ** i for i in range(2, degree + 1)]))
# Make the prediction:
predictions = model1.predict(new_X_poly)
print(model1.params)
fig2, ax2 = plt.subplots(figsize=(6, 4))
sm.graphics.plot_fit(model1, 1, ax=ax2)
ax2.set_xlabel('Year')
plt.title('Installed Solar Energy Capacity', color='blue')
plt.grid()
plt.show()
print("Predictions:", list(zip(new_X, predictions)))

# Using the polynomial model for regression:
# Polynomial features:
degree = 2
X_poly = sm.add_constant(np.column_stack([df_shr['Year']] + [df_shr['Year'] ** i for i in range(2, degree + 1)]))
# Fit the polynomial model:
model1 = sm.OLS(df_shr['Solar - % electricity'], X_poly).fit()
# New data points for prediction


def create_year_array(start_year, end_year):
    return np.array([year for year in range(start_year, end_year + 1)])


new_X = create_year_array(2022, 2030)
# Polynomial features for the new data:
new_X_poly = sm.add_constant(np.column_stack([new_X] + [new_X ** i for i in range(2, degree + 1)]))
# Make the prediction:
predictions = model1.predict(new_X_poly)
print(model1.params)
fig2, ax2 = plt.subplots(figsize=(6, 4))
sm.graphics.plot_fit(model1, 1, ax=ax2)
ax2.set_xlabel('Year')
plt.title('Share of Electricity Production From Solar', color='blue')
plt.grid()
plt.show()
print("Predictions:", list(zip(new_X, predictions)))
