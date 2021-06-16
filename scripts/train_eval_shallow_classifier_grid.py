from reflection_classification.utils.dataset import ReflexiveDataset
from reflection_classification.shallow_classifier import ShallowClassifier

from sklearn.metrics import accuracy_score
import argparse

cfrs = ['random_forrest', 'logistic_regression', 'naive_bayes', 'support_vector_classifier']


cfr_best = {cfr: {train_val: {test_val: 0 for test_val in range(3, 7)} for train_val in range(3, 7)} for cfr in cfrs}

if __name__ == "__main__":
    # print("\t".join(["classifier", "context", "train confidence", "test confidence", "accuracy"]))
    for vocab_size in range(100, 1000, 50):
        # print("Context: %s" % vocab_size)
        for train_conf in range(3, 7):
            # print("Train conf: %s" % train_conf)
            for test_conf in range(3, 7):
                # print("Test conf: %s" % test_conf)
                for cfr_name in cfrs:
                    # print("Classifier: %s" % cfr_name)
                    argparser = argparse.ArgumentParser()

                    argparser.add_argument('--classifier', type=str,
                                           help='Classifier to use. One of: {random_forrest, logistic_regression, '
                                                'naive_bayes, support_vector_classifier}', default=cfr_name)
                    argparser.add_argument('--sentences_dir', type=str, required=True,
                                           help='Directory with {split}/sentence.tsv of annotated sentences')
                    argparser.add_argument('--train_confidence_threshold', type=int,
                                           help='Minimal confidence threshold for sentences to train on.',
                                           default=train_conf)
                    argparser.add_argument('--test_confidence_threshold', type=int,
                                           help='Minimal confidence threshold for sentences to test on.',
                                           default=test_conf)
                    argparser.add_argument('--use_context', type=bool, help='Whether the model was trainer using context.',
                                           default=True)
                    argparser.add_argument('--lang', type=bool, help='Whether the model was trainer using context.',
                                           default="cze")
                    argparser.add_argument('--vocabulary_size', type=int,
                                           help='Number of top-n most-occurring words used '
                                                'to create Bag of Words representation for classification',
                                           default=vocab_size)
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

                    cfr = ShallowClassifier(classifier=classifier, use_context=args.use_context,
                                            bow_size=args.vocabulary_size, lang=args.lang)
                    cfr.train(train_sentences)
                    pred_targets = cfr.predict(test_sentences)
                    true_targets = [s.label for s in test_sentences]
                    objective_val = accuracy_score(true_targets, pred_targets)
                    # print("Classification report: \n%s" % classification_report(true_targets, pred_targets))
                    print("\t".join(map(str, [args.classifier, vocab_size, train_conf, test_conf, objective_val])))

                    if objective_val > cfr_best[cfr_name][train_conf][test_conf]:
                        cfr_best[cfr_name][train_conf][test_conf] = objective_val

print(cfr_best)

print()
