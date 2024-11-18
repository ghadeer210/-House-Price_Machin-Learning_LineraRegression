from lib import *

# Importing the DataSet

# dataset = pd.read_csv("Path_DataSet like ---> .../file.csv")
dataset = pd.read_csv(r'C:\Users\Ghadeer\Desktop\House Prices ML&Git\DataSet\train.csv')

# DropOut mamy columns
#[ 
# 'Id', 'Alley', 'MasVnrType',
# 'BsmtQual', 'BsmtCond', 'BsmtExposure',
# 'BsmtFinType1', 'BsmtFinType2', 'Electrical',
# 'FireplaceQu', 'GarageType', 'GarageFinish',
# 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 
# ]

dataset = dataset.drop(['Id', 'Alley', 'MasVnrType',
                        'BsmtQual', 'BsmtCond', 'BsmtExposure',
                        'BsmtFinType1', 'BsmtFinType2', 'Electrical',
                        'FireplaceQu', 'GarageType', 'GarageFinish',
                        'GarageQual', 'GarageCond', 'PoolQC',
                        'Fence', 'MiscFeature', 
                        ],axis=1)

## Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
for i in [2, 23, 48]:
    column_name = dataset.columns[i]
    imputer.fit(dataset[[column_name]])
    dataset[[column_name]] = imputer.transform(dataset[[column_name]])
    
# Dictionary to store column labels
column_labels = {}
column_types_numeric = []
column_types_object = []

# Loop through columns and assign index as label
for idx, column in enumerate(dataset.columns):
    column_labels[column] = idx
    # Check if the column is numeric
    if pd.api.types.is_numeric_dtype(dataset[column]):
        column_types_numeric.append(column_labels[column])
    else:
        column_types_object.append(column_labels[column])
        
# Encoder categorical data
X = dataset.iloc[:, :-1]
for name_indx in column_types_object:
    label_encoder = LabelEncoder()
    label_encoder.fit(X.iloc[:, name_indx])
    X[X.columns[name_indx]] = label_encoder.transform(X.iloc[:, name_indx])

X = X.values
y = dataset.iloc[:, -1].values
# Split DataSet after prepare data
X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y,
                                                    test_size = 0.2,
                                                    random_state = 1
                                                    )
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###########################################
##########Function to run this file#########
def run_init_Data(path:str, test_size:float=0.2, random_state:int=1):
    dataset = pd.read_csv(path)
    test_size = test_size
    random_state = random_state
    
    dataset = dataset.drop(['Id', 'Alley', 'MasVnrType',
                        'BsmtQual', 'BsmtCond', 'BsmtExposure',
                        'BsmtFinType1', 'BsmtFinType2', 'Electrical',
                        'FireplaceQu', 'GarageType', 'GarageFinish',
                        'GarageQual', 'GarageCond', 'PoolQC',
                        'Fence', 'MiscFeature', 
                        ],axis=1)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    for i in [2, 23, 48]:
        column_name = dataset.columns[i]
        imputer.fit(dataset[[column_name]])
        dataset[[column_name]] = imputer.transform(dataset[[column_name]])
        
        column_labels = {}
        column_types_numeric = []
        column_types_object = []

        for idx, column in enumerate(dataset.columns):
            column_labels[column] = idx
            if pd.api.types.is_numeric_dtype(dataset[column]):
                column_types_numeric.append(column_labels[column])
            else:
                column_types_object.append(column_labels[column])
                
        X = dataset.iloc[:, :-1]
        for name_indx in column_types_object:
            label_encoder = LabelEncoder()
            label_encoder.fit(X.iloc[:, name_indx])
            X[X.columns[name_indx]] = label_encoder.transform(X.iloc[:, name_indx])

        X = X.values
        y = dataset.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(
                                                            X, y,
                                                            test_size = 0.2,
                                                            random_state = 1
                                                            )
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        return X_train, y_train

