import pandas as pd

def load_and_clean_data(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Numerical columns 
    num_cols = [
        'year_sold', 'property_tax', 'insurance', 'beds', 'baths',
        'sqft', 'year_built', 'lot_size', 'property_age', 'price'
    ]

    # Binary columns 
    bin_cols = ['basement', 'popular', 'recession', 'property_type_Condo']

    # Fill missing values for numerical features with median
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill missing values for binary features with mode
    for col in bin_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df