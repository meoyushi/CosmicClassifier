# CosmicClassifier

## Importing Libraries : The notebook begins with essential imports and initial dataset loading.

## Exploratory Data Analysis
- **Loading the Dataset** : df = pd.read_csv("thermoracleTrain.csv") – Reads a CSV file (thermoracleTrain.csv) into a Pandas DataFrame named df.
- **Displaying Column Names** : This prints the names of all columns in the dataset.
- **Dataset Summary** : Mean for all columns (except Prediction) is almost zero indicating the data is scaled. Also there is presence of two categorical columns- Magnetic Field Strength and Radiation Levels.

## Dealing with Missing values in Numerical Columns
- **Drop columns where Prediction is Null.** : Re-checks for missing values after handling them. Ideally, the output should now be zero for all columns.
- **Filling the missing values in rest of numerical columns with the median of the respective columns**
- **Fairly Balanced** : The data is fairly balanced since no single class dominates significantly.

## Categorical Columns Labeling and Cleaning
