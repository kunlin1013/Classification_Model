import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(csvpath: str, split_ratio: float=0.3):
    df_csv = pd.read_csv(csvpath)
    train_set = df_csv[df_csv['data set'] == "train"]

    train_data, val_data = train_test_split(train_set, test_size=0.3, random_state=42)
    
    # Find the maximum number of each category
    class_counts = train_data['class id'].value_counts()
    max_count = class_counts.max()

    balanced_train_data = pd.DataFrame()
    for class_id, _ in class_counts.items():
        class_data = train_data[train_data['class id'] == class_id]
        
        # Calculate the number of resamples needed
        resample_count = max_count - class_data.shape[0]
        
        # Perform resampling if necessary
        if resample_count > 0:
            resampled_data = class_data.sample(n=resample_count, replace=True, random_state=42)
            class_data_balanced = pd.concat([class_data, resampled_data])
        else:
            class_data_balanced = class_data
        
        balanced_train_data = pd.concat([balanced_train_data, class_data_balanced])
        
    # Convert to dictionaries
    balanced_train_data = balanced_train_data.to_dict('list')
    val_data = val_data.to_dict('list')
    
    return balanced_train_data, val_data
