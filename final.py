import pandas as pd
import numpy as np
import re
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import math
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
scoring_metrics = ["accuracy", "precision", "recall", "f1"]

best_feature_name_path = "./feature_names.json"
dataset_path = "./IMDB Dataset.csv"


def generate_bigrams(word_list):
    bigrams_list = []
    for ngram in ngrams(word_list, 2):
        bigrams_list.append(" ".join(ngram))
    return bigrams_list


def clean_raw_text(text: str):
    # most of the cleaning is done by sklearn's vectorizer
    text = text.lower().replace(r"<br /><br />", " ")
    return text


def remove_df_duplicates(df, col_name):
    df_no_duplicates = df[~df["review"].duplicated(keep="first")]
    return df_no_duplicates


def write_best_feature_names(feature_names):
    with open(best_feature_name_path, "w") as file:
        json.dump(feature_names.tolist(), file)

    return feature_names


def load_best_feature_names():
    with open(best_feature_name_path, "r") as file:
        feature_names = json.load(file)

    return feature_names


def get_matrix(df, stem=None, stop=None, ngrams=1):
    reviews = df["review"].tolist()
    analyzer = TfidfVectorizer().build_analyzer()

    def stemmed_words(doc):
        temp = analyzer(doc)
        if stop:
            temp = [word for word in temp if word not in stop_words]
        if stem == "porter":
            temp = [stemmer.stem(word) for word in temp]

        if ngrams > 1:
            temp = temp + generate_bigrams(temp)

        return temp

    vectorizer = TfidfVectorizer(
        # ngram_range=(1, 2),
        # stop_words=list(stop_words),
        analyzer=stemmed_words,
    )
    tfidf_matrix = vectorizer.fit_transform(reviews)

    return tfidf_matrix, vectorizer.get_feature_names_out()


def train_and_eval(clf, data, labels):
    cross_val_results = cross_validate(
        clf,
        data,
        labels,
        cv=5,
        scoring=scoring_metrics,
        n_jobs=-1,
    )

    return cross_val_results


def save_results(cross_val_results, name):
    out_data = {}
    for metric, scores in cross_val_results.items():
        out_data[metric] = {"scores": scores.tolist(), "mean": scores.mean()}

    with open(name, "w") as file:
        json.dump(out_data, open(name, "w"))


# takes in a preprocessed dataframe and returns a matrix of TF-IDF features
# using the parameters passed in
def prepare_dataset(df, stem=None, stop=None, ngrams=1, feature_val=3_000):
    labels = df["sentiment"].tolist()
    matrix, feature_names = get_matrix(df, stem=stem, stop=stop, ngrams=ngrams)
    num_features = matrix.shape[1]
    selector = SelectKBest(chi2, k=feature_val)

    X_new = selector.fit_transform(matrix, labels)
    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = feature_names[selected_feature_indices]

    return X_new, labels, selected_feature_names


def bulk_process(data_packs):
    df = pd.read_csv(dataset_path)

    df = remove_df_duplicates(df, "review")
    # clean each row
    df.loc[:, "review"] = df["review"].apply(clean_raw_text)
    df.loc[:, "sentiment"] = df["sentiment"].apply(
        lambda a: 1 if a == "positive" else 0
    )

    for i in data_packs:
        classifier_type, stop_val, stem_val, ngrams, feature_val = i

        gram_translation = (
            "unigram" if ngrams == 1 else "bigram" if ngrams == 2 else "trigram"
        )
        stop_translation = "stop" if stop_val else "nostop"
        name = f"{classifier_type}_{gram_translation}_{stem_val}_{stop_translation}_{feature_val}.json"
        clf = (
            SVC(kernel="linear", random_state=42)
            if classifier_type == "svc"
            else LogisticRegression(random_state=42, max_iter=1000)
        )

        # start of actual work
        data, labels, feature_names = prepare_dataset(
            df, stem=stem_val, stop=stop_val, ngrams=ngrams, feature_val=feature_val
        )
        results = train_and_eval(clf, data, labels)
        save_results(results, name)


def process_all_permutations():
    # a number of permutations of the data to test
    # in format, [classifier, stop, stem, ngrams, chi square feature val]
    vals = [
        ["lr", False, None, 1, 3_000],
        ["lr", True, None, 1, 3_000],
        ["lr", False, "porter", 1, 3_000],
        ["lr", True, "porter", 1, 3_000],
        ["lr", False, None, 2, 10_000],
        ["lr", True, None, 2, 10_000],
        ["lr", False, "porter", 2, 10_000],
        ["lr", True, "porter", 2, 10_000],
        ["svc", False, None, 1, 3_000],
        ["svc", True, None, 1, 3_000],
        ["svc", False, "porter", 1, 3_000],
        ["svc", True, "porter", 1, 3_000],
        ["svc", False, None, 2, 10_000],
        ["svc", True, None, 2, 10_000],
        ["svc", False, "porter", 2, 10_000],
        ["svc", True, "porter", 2, 10_000],
    ]

    bulk_process(vals)


## start of evaluation


# load the set of features processed with the best parameters as determined by the initial search
def get_best_processed_data():
    stop_val = False
    stem_val = None
    ngrams = 2
    feature_val = 10_000

    df = pd.read_csv(dataset_path)

    df = remove_df_duplicates(df, "review")
    # clean each row
    df.loc[:, "review"] = df["review"].apply(clean_raw_text)
    df.loc[:, "sentiment"] = df["sentiment"].apply(
        lambda a: 1 if a == "positive" else 0
    )

    # start of actual work
    data, labels, feature_names = prepare_dataset(
        df, stem=stem_val, stop=stop_val, ngrams=ngrams, feature_val=feature_val
    )

    return data, labels, feature_names


def write_param_grid_result(results, name, print_output=False):
    with open(name, "w") as file:
        output = []
        for i in range(len(results["params"])):
            params = results["params"][i]
            accuracy = round(results["mean_test_accuracy"][i], 4)
            precision = round(results["mean_test_precision"][i], 4)
            recall = round(results["mean_test_recall"][i], 4)
            f1 = round(results["mean_test_f1"][i], 4)

            out_dict = {
                "parameters": params,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "fscore": f1,
            }

            if print_output:
                print(out_dict)
            output.append(out_dict)

        json.dump(output, file)


def param_grid_test(clf, param_grid, data, labels, name):
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        scoring=scoring_metrics,
        n_jobs=-1,
        verbose=1,
        refit=False,
    )
    grid_search.fit(data, labels)

    results = grid_search.cv_results_
    write_param_grid_result(results, name, print_output=True)


def test_best_svc():
    data, labels, feature_names = get_best_processed_data()
    param_grid = {
        "C": [1, 5, 10, 25],
        "kernel": ["linear", "rbf"],
    }

    svc_model = SVC(random_state=42, shrinking=True)
    param_grid_test(svc_model, param_grid, data, labels, "best_svc_results.json")


def test_best_lr():
    data, labels, feature_names = get_best_processed_data()
    param_grid = {
        "C": [1, 5, 10, 25],
    }

    logreg = LogisticRegression(random_state=42, max_iter=1000)
    param_grid_test(logreg, param_grid, data, labels, "best_lr_results.json")


def get_best_svc():
    data, labels, feature_names = get_best_processed_data()

    svc_model = SVC(random_state=42, shrinking=True, C=5, kernel="rbf")
    svc_model.fit(data, labels)

    return svc_model, feature_names


def get_best_lr():
    data, labels, feature_names = get_best_processed_data()

    logreg = LogisticRegression(random_state=42, max_iter=1000, C=25)
    logreg.fit(data, labels)

    return logreg, feature_names


## word cloud implementation


def get_lr_weights(clf):
    weights = clf.coef_[0]
    return weights


def write_lr_weights(clf, name):
    weights = get_lr_weights(clf)
    with open(name, "w") as file:
        json.dump(weights.tolist(), file)

    return weights


def load_weights(name):
    with open(name, "r") as file:
        weights = json.load(file)

    return weights


def get_high_polarity_words(feature_names, feature_weights):
    features_df = pd.DataFrame({"feature": feature_names, "weight": feature_weights})

    # top_features = features_df.sort_values(by="weight", ascending=False)

    highest_values = features_df.iloc[
        (features_df["weight"]).abs().argsort()[::-1]
    ].head(100)
    lowest_values = features_df.iloc[(features_df["weight"]).abs().argsort()].head(100)

    return highest_values, lowest_values


def make_word_cloud(weights_df, name):
    weights_dict = weights_df.set_index("feature")["weight"].to_dict()
    abs_dict = {k: abs(v) for k, v in weights_dict.items()}
    # sqrt values to make the word cloud more readable
    abs_dict = {k: math.sqrt(math.sqrt(v)) for k, v in abs_dict.items()}

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        sentiment = weights_dict.get(word)
        if sentiment > 0:
            return "blue"
        elif sentiment < 0:
            return "red"

        return "black"

    wordcloud = WordCloud(
        random_state=42,
        width=800,
        height=400,
        background_color="white",
        color_func=color_func,
    ).generate_from_frequencies(abs_dict)

    plt.figure(figsize=(10, 5))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")
    plt.savefig(name)


if __name__ == "__main__":
    # initial evaluation
    process_all_permutations()

    # final evaluation
    test_best_lr()
    test_best_svc()
    get_best_lr()
    get_best_svc()

    # calculate first time
    best_lr, _ = get_best_lr()
    best_svc, _ = get_best_svc()

    # replace to switch between cached and calculated
    _, _, feature_names = get_best_processed_data()
    write_best_feature_names(feature_names)

    feature_names = load_best_feature_names()

    best_lr_weights = get_lr_weights(best_lr)
    write_lr_weights(best_lr, "lr_weights.json")
    best_lr_weights = load_weights("lr_weights.json")

    highest_lr, lowest_lr = get_high_polarity_words(feature_names, best_lr_weights)

    # word cloud
    make_word_cloud(highest_lr, "word_cloud.png")
