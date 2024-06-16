import pandas as pd

def main():
    # load the dataset that is in path: Assignment-2/result.csv
    df = pd.read_csv("results.csv")
    # data frame manipulation

    # the first line in the df is the logistic regression
    logistic = df.iloc[0]

    activations = ["tanh", "sigmoid", "relu"]
    dfs = []

    # add the logistic regression to the df for each pivot for each activation
    for activation in activations:
        df_activation = df[df["activation"] == activation]

        # Create a pivot table for test_accuracy
        df_test = df_activation.pivot(index='hidden_size', columns='epochs', values='test_accuracy')
        df_test.loc[0] = logistic["test_accuracy"]

        # Create a pivot table for train_accuracy
        df_train = df_activation.pivot(index='hidden_size', columns='epochs', values='train_accuracy')
        df_train.loc[0] = logistic["train_accuracy"]

        dfs.append((activation, df_train, df_test))
    
    return dfs

if __name__ == "__main__":
    main()