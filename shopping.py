import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Name: Côme Sören Noé 
# Surname: NZUZI MANIEMA
# Student ID: 240 AMM 031

TEST_SIZE = 0.4


def main():
    print("Script path and name:", sys.argv[0])
    print("Number of arguments:", len(sys.argv))
    print("Argument List:", str(sys.argv))

    # Check command-line arguments
    if len(sys.argv) == 1:
        filename = "c:\\Users\\NZUZI MANIEMA\\Documents\\AERO 4\\Riga Semestre 7\\Cours\\Dev of App Systems\\Learning\\shopping.csv"
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        sys.exit("Usage: python shopping.py data")
    
    print(f"Using file: {filename}")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(filename)
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    months= {"Jan":0, "Feb":1, "Mar":2, "Apr":3, "May":4, "June":5, "Jul":6, "Aug":7, "Sep":8, "Oct":9, "Nov":10, "Dec":11}
    evidence =[] # evidence corresponds to the features
    labels = [] # labels corresponds to the target

    with open(filename, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            evidence.append([
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                months[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0
            ])
            labels.append(1 if row["Revenue"] == "TRUE" else 0)
    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=10) # define the model with k-value
    print("k value: ", model.n_neighbors)
    model.fit(evidence, labels) # fit the model (<=> train the model)
    return model # return the model trained



def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    
    # True positive rate (sensitivity)
    true_positive =sum(1 for actual, predicted in zip(labels, predictions) if actual == 1 and predicted == 1)
    total_positive = sum(1 for actual in labels if actual == 1)
    sensitivity = true_positive / total_positive if total_positive > 0 else 0.0
    true_negative =sum(1 for actual, predicted in zip(labels, predictions) if actual == 0 and predicted == 0)
    total_negative = sum(1 for actual in labels if actual == 0)
    specificity = true_negative / total_negative if total_negative > 0 else 0.0

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()

"""


** For k=1 **

Correct: 4042
Incorrect: 890
True Positive Rate: 37.87%
True Negative Rate: 89.76%
(but the results may vary)

* Output interpretation:
- Correct: 4042 
This shows the total number of predictions that our classifier got correct. Specifically, it includes both true positives (correctly predicting that a user will make a purchase) and true negatives (correctly predicting that a user will not make a purchase).
Thus, out of all the test predictions, 4 042 were correct.

- Incorrect: 890
However, this is the number of predictions that our classifier got wrong. More precisely , it includes both false positives (predicting a purchase when there isn't one) and false negatives (predicting no purchase when there is one).
So, 890 predictions were wrong out of the total test set.

- True Positive Rate: 37.87%
Sensitivity, also known as the true positive rate, measures how well the classifier correctly identifies users who actually make a purchase. It is the proportion of actual purchasers (positive labels) that are correctly identified.
Sensitivity = True Positives / Total Positives with True Positives which are the users who actually made a purchase and were correctly predicted as such. Also, Total Positives are all users who actually made a purchase.
39.37% of users who made a purchase were correctly identified by our model. This indicates that our model is not doing a great job identifying positive instances (users who make purchases). A higher sensitivity would be better, especially if the goal is to target potential buyers effectively.

- True Negative Rate: 89.76%
Specificity, also known as the true negative rate, measures how well the classifier correctly identifies users who do not make a purchase. It is the proportion of actual non-purchasers (negative labels) that are correctly identified.
Specificity = True Negatives / Total Negatives with True Negatives which are the users who did not make a purchase and were correctly predicted as such. Also, Total Negatives are all users who did not make a purchase.
89.76% of users who did not make a purchase were correctly identified by our model. This means that our model is doing a good job identifying negative instances (users who do not make purchases). A higher specificity would be better, especially if the goal is to avoid targeting users who are unlikely to make a purchase.

To conclude, our model is quite good at predicting users who won't make a purchase. It avoids incorrectly classifying non-purchasers as buyers.
Nevertheless, the model struggles to identify users who will make a purchase. This could mean that it misses many potential buyers (false negatives).

I know it was not asked but I changed the k value to see how it affects the model's performance.
Let's delve into the results.

** For k=2 **

- Correct: 4227
- Incorrect: 705
- True Positive Rate: 21.69% our model is now even worse at identifying users who will make a purchase.
- True Negative Rate: 97.29% The model is now much better at identifying users who will not make a purchase.

By increasing k from 1 to 2, the model is focusing more on correctly identifying users who will not make a purchase (high specificity), but it is struggling even more to identify actual buyers (low sensitivity). 
The downside is that minority class predictions (purchasers) become harder to make correctly.

** For k=10 **

- Correct: 4271 better overall accuracy.
- Incorrect: 661 The number of incorrect predictions has decreased which indicates the model is making fewer errors.
- True Positive Rate: 17.65% The model is now even less capable of identifying users who will make a purchase. 
- True Negative Rate: 99.47% Specificity has reached an exceptionally high level. The model is now correctly identifying almost all non-purchasers with very few false positives.

To conclude, our model is becoming increasingly biased toward the majority class (non-purchasers), as evident by the exceptionally high specificity.
Then, with k=10, our model is fantastic at identifying non-purchasers but performs poorly at detecting purchasers. 

"""
