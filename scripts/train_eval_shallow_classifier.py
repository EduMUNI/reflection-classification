from reflection_classification.utils.dataset import ReflexiveDataset
from reflection_classification.shallow_classifier import ShallowClassifier

from sklearn.metrics import f1_score, classification_report
import argparse


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--classifier', type=str,
                           help='Classifier to use. One of: {random_forrest, logistic_regression, '
                                'naive_bayes, support_vector_classifier}', required=True)
    argparser.add_argument('--sentences_dir', type=str, required=True,
                           help='Directory with {split}/sentence.tsv of annotated sentences')
    argparser.add_argument('--train_confidence_threshold', type=int,
                           help='Minimal confidence threshold for sentences to train on.',
                           default=5)
    argparser.add_argument('--test_confidence_threshold', type=int,
                           help='Minimal confidence threshold for sentences to test on.',
                           default=5)
    argparser.add_argument('--use_context', type=bool, help='Whether the model was trainer using context.',
                           default=True)
    argparser.add_argument('--language', type=str, help='Language to decide on how to remove stemming.',
                           default="en")
    argparser.add_argument('--vocabulary_size', type=int,
                           help='Number of top-n most-occurring words used '
                                'to create Bag of Words representation for classification',
                           default=300)
    args = argparser.parse_args()

    if args.classifier == 'random_forrest':
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier()
    elif args.classifier == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(max_iter=10e4)
    elif args.classifier == 'naive_bayes':
        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB()
    elif args.classifier == 'support_vector_classifier':
        from sklearn.svm import SVC
        classifier = SVC()
    else:
        raise ValueError("Unrecognized classifier: %s" % args.classifier)

    train_sentences = ReflexiveDataset.sentences_from_tsv(args.sentences_dir, "train",
                                                          args.train_confidence_threshold, args.use_context)
    test_sentences = ReflexiveDataset.sentences_from_tsv(args.sentences_dir, "test",
                                                         args.test_confidence_threshold, args.use_context)

    cfr = ShallowClassifier(classifier=classifier, use_context=args.use_context, bow_size=args.vocabulary_size,
                            lang=args.language)
    cfr.train(train_sentences)
    pred_targets = cfr.predict(test_sentences)
    true_targets = [s.label for s in test_sentences]
    objective_val = f1_score(true_targets, pred_targets, average='micro')
    print("Evaluating on %s sentences" % len(test_sentences))
    print("Classification report: \n%s" % classification_report(true_targets, pred_targets))
    print(objective_val)
