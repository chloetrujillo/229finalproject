import pandas as pd


def main():
    val_df = pd.read_csv("data/val.csv")
    test_df = pd.read_csv("data/test.csv")

    # Majority class in train dataset is 8 stars
    MAJORITY_CLASS = 8
    
    # Print out total number of levels in val dataset
    print(val_df.shape[0])

    # Print out the number of levels for 7, 8, 9 stars in val dataset
    print(val_df[val_df["stars"] == 7].shape[0])
    print(val_df[val_df["stars"] == 8].shape[0])
    print(val_df[val_df["stars"] == 9].shape[0])

    # Calculate accuracy and one-off accuracy on val dataset
    val_len = val_df.shape[0]
    got_right = val_df[val_df["stars"] == MAJORITY_CLASS].shape[0]
    one_off = got_right + val_df[val_df["stars"] == MAJORITY_CLASS + 1].shape[0] + val_df[val_df["stars"] == MAJORITY_CLASS - 1].shape[0]
    val_accuracy = got_right / val_len
    val_one_off_accuracy = one_off / val_len
    print(f"Val Accuracy: {val_accuracy}, Val One-Off Accuracy: {val_one_off_accuracy}")


    # Print out total number of levels in test dataset
    print(test_df.shape[0])

    # Print out the number of levels for 7, 8, 9 stars in test dataset
    print(test_df[test_df["stars"] == 7].shape[0])
    print(test_df[test_df["stars"] == 8].shape[0])
    print(test_df[test_df["stars"] == 9].shape[0])

    # Calculate accuracy and one-off accuracy on test dataset
    test_len = test_df.shape[0]
    got_right = test_df[test_df["stars"] == MAJORITY_CLASS].shape[0]
    one_off = got_right + test_df[test_df["stars"] == MAJORITY_CLASS + 1].shape[0] + test_df[test_df["stars"] == MAJORITY_CLASS - 1].shape[0]
    test_accuracy = got_right / test_len
    test_one_off_accuracy = one_off / test_len
    print(f"Test Accuracy: {test_accuracy}, Test One-Off Accuracy: {test_one_off_accuracy}")

if __name__ == "__main__":
    main()