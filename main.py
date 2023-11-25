import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

def main():
    dataset = datasets.load_breast_cancer()

    # Write the dataset to an Excel file
    df = pd.DataFrame(data=dataset, columns=dataset.feature_names)
    df['target'] = dataset.target
    
    excel_file_path = "breast_cancer_dataset.xlsx"
    df.to_excel(excel_file_path, index=False)

if __name__ == '__main__':
    main()
