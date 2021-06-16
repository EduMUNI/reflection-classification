import argparse
from tqdm import tqdm

from reflection_classification.utils.dataset import ReflexiveDataset
from reflection_classification.neural_classifier import NeuralClassifier

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--trained_model_dir', type=str, required=True,
                           help='Local path containing pre-trained model, filled on training, or downloaded separately')
    argparser.add_argument('--sentences_dir', type=str, required=True,
                           help='Directory with {split}/sentence.tsv of annotated sentences')
    argparser.add_argument('--device', type=str, help='Device used to infer. One of {cpu, cuda, cuda:[idx]}',
                           default="cuda")
    argparser.add_argument('--test_confidence_threshold', type=int,
                           help='Minimal confidence threshold for sentences to test on.',
                           default=5)
    argparser.add_argument('--use_context', type=bool, help='Whether the model was trainer using context.',
                           default=True)

    args = argparser.parse_args()

    classifier = NeuralClassifier(args.trained_model_dir, args.use_context, args.device)
    test_sentences = ReflexiveDataset.sentences_from_tsv(args.sentences_dir, "test",
                                                         args.test_confidence_threshold, args.use_context)

    y_pred = [classifier.predict_sentence(sentence.text, sentence.context) for sentence in tqdm(test_sentences)]

    y_trues = [sentence.label for sentence in test_sentences]

    y_truepos = [y_trues[i] == y_pred[i] for i, _ in enumerate(y_pred)]

    print("Test accuracy: %s" % (sum(y_truepos) / len(y_truepos)))
