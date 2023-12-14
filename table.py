import pandas as pd
import matplotlib.pyplot as plt
import json

def process_table_values(x):
    print(x)
    if type(x) == float:
        return  round(x, 4)

    return x

def create_table(data, name):


    df = pd.DataFrame(data)
    df = df.applymap(process_table_values)

    df.loc[:, "model"] = df["model"].apply(lambda name: name.replace("me", "lr"))

    df.set_index('model', inplace=True)

    df.sort_values(by=['fscore'], inplace=True, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values,
            colLabels=df.columns,
             rowLabels=df.index,
            cellLoc='center',
            loc='center')

    plt.savefig(name, bbox_inches='tight', pad_inches=0.5)



def parse_initial_experiment_files(filenames):
    data = {
            'model': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'fscore': [],
}
    for filename in filenames:
        with open(filename, 'r') as f:
            file_data = json.load(f)

        end_stripped = filename.split('.')[0]
        parts = end_stripped.split('_')
        model = parts[1]
        gram = parts[2]
        stem = parts[3]
        stop = parts[4]

        if stop == "nostop":
            stop = "no-stop"
        if stem == "None":
            stem = "no-stem"

        model_descriptor = f"{model} {gram} {stem} {stop}"
        data["model"].append(model_descriptor)
        data["accuracy"].append(file_data["test_accuracy"]["mean"])
        data["precision"].append(file_data["test_precision"]["mean"])
        data["recall"].append(file_data["test_recall"]["mean"])
        data["fscore"].append(file_data["test_f1"]["mean"])

    return data

def parse_refine_experiment_files():    
    combined_data = []
    for clf in ["lr", "svc"]:
        data = json.load(open(f"best_{clf}_results.json", 'r'))
        for d in data:
            d["model"] = f"{clf} {d['parameters']}"
            del d["parameters"]
        combined_data.extend(data)

    return combined_data


if __name__ == "__main__":

    # initial experiment
    input_filenames = [
        "testing2_me_bigram_None_nostop_10000.json",
        "testing2_me_bigram_None_stop_10000.json",
        "testing2_me_bigram_porter_nostop_10000.json",
        "testing2_me_bigram_porter_stop_10000.json",
        "testing2_me_unigram_None_nostop_3000.json",
        "testing2_me_unigram_None_stop_3000.json",
        "testing2_me_unigram_porter_nostop_3000.json",
        "testing2_me_unigram_porter_stop_3000.json",
        "testing2_svc_bigram_None_nostop_10000.json",
        "testing2_svc_bigram_None_stop_10000.json",
        "testing2_svc_bigram_porter_nostop_10000.json",
        "testing2_svc_bigram_porter_stop_10000.json",
        "testing2_svc_unigram_None_nostop_3000.json",
        "testing2_svc_unigram_None_stop_3000.json",
        "testing2_svc_unigram_porter_nostop_3000.json",
        "testing2_svc_unigram_porter_stop_3000.json"
    ]
    data = parse_initial_experiment_files(input_filenames)
    create_table(data, "model_results.png")


    # # refined experiment
    # data = parse_refine_experiment_files()
    # create_table(data, "refined_results.png")
    # print(data)