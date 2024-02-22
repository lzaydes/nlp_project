# TODO - import relevant sklearn score modules 
from sklearn.metrics import f1_score, accuracy_score
import argparse
from src.utils.file_utils import load_jsonl

def evaluate_standard(gt_labels, pred_labels):
    
    accuracy, f1score = 0, 0

    ##################################################
    # TODO: Please finish the standard evaluation metrics.
    # You need to compute the accuracy and F1 score for the 
    # predictions and ground truth labels. Please use the 
    # scikit-learn APIs in way they can deal with strings 
    # as label. Remeber to import the functions you use!

    gt_labels_encoded = []
    pred_labels_encoded = []

    for each in gt_labels:
        if each == "SUPPORTS":
            gt_labels_encoded.append(1)
        elif each == "REFUTES":
            gt_labels_encoded.append(0)
        else:
            print(f'Invalid label in gt labels: {each}')

    for each in pred_labels:
        if each == "SUPPORTS":
            pred_labels_encoded.append(1)
        elif each == "REFUTES":
            pred_labels_encoded.append(0)
        else:
            print(f'Invalid label in predicted labels: {each}')

    accuracy = accuracy_score(gt_labels_encoded, pred_labels_encoded)
    f1score = f1_score(gt_labels_encoded, pred_labels_encoded)

    # End of TODO.
    ##################################################

    return accuracy, f1score

def model_eval_report(gt_filepath, pred_filepath):
    
    gt_data = load_jsonl(gt_filepath)
#    print("Loaded training predictions successfully")
    gt_labels = [d["label"] for d in gt_data]
   # pred_data = load_jsonl(pred_filepath)
    #pred_labels = [d["label"] for d in pred_data]
    with open(pred_filepath, "r") as f:
        pred_labels = [d.strip() for d in f.readlines()]
    accuracy, f1score = evaluate_standard(gt_labels, 
                                          pred_labels)

    print(f"Overall Accuracy : {accuracy}")
    print(f"Overall F1 score : {f1score}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_filepath', type=str, required=True)
    parser.add_argument('--pred_filepath', type=str, required=True)

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    args = parse_args()
    model_eval_report(args.gt_filepath, args.pred_filepath)
