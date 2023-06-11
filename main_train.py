import os
import yaml
import glob
import numpy

import pickler
import generate_data

from tqdm import tqdm
from sklearn import metrics
from visualizer import visualizer
from model import keras_model
from logger import logger

# load parameter yaml
with open("./config/directory_config.yaml") as stream:
    dir_param = yaml.safe_load(stream)

with open("./config/model_config.yaml") as stream:
    model_param = yaml.safe_load(stream)


# make output directory
if not os.path.exists(dir_param["root_dir"]):
    os.makedirs("{root}".format(root=dir_param["root_dir"]), exist_ok=True)

    os.makedirs(
        "{root}{dir}".format(root=dir_param["root_dir"], dir=dir_param["pickle_dir"]),
        exist_ok=True,
    )
    os.makedirs(
        "{root}{dir}".format(root=dir_param["root_dir"], dir=dir_param["pickle_dir"]),
        exist_ok=True,
    )
    os.makedirs(
        "{root}{dir}".format(root=dir_param["root_dir"], dir=dir_param["model_dir"]),
        exist_ok=True,
    )
    os.makedirs(
        "{root}{dir}".format(root=dir_param["root_dir"], dir=dir_param["result_dir"]),
        exist_ok=True,
    )

# load base_directory list
dirs = sorted(
    glob.glob(
        os.path.abspath(".{dataset}/*/*/*".format(dataset=dir_param["dataset_dir"]))
    )
)

# setup the result
result_file = "{root}/{result_dir}/{result_file}".format(
    root=dir_param["root_dir"],
    result_dir=dir_param["result_dir"],
    result_file=dir_param["result_file"],
)
results = {}

# loop of the dataset directory
for dir_idx, target_dir in enumerate(dirs):
    print(
        "[{num}/{total}] {dirname}".format(
            dirname=target_dir, num=dir_idx + 1, total=len(dirs)
        )
    )

    # dataset param
    db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
    machine_type = os.path.split(os.path.split(target_dir)[0])[1]
    machine_id = os.path.split(target_dir)[1]

    # setup path
    evaluation_result = {}
    train_pickle = (
        "{root}/{pickle}/train_{machine_type}_{machine_id}_{db}.pickle".format(
            root=dir_param["root_dir"],
            pickle=dir_param["pickle_dir"],
            machine_type=machine_type,
            machine_id=machine_id,
            db=db,
        )
    )

    eval_files_pickle = (
        "{root}/{pickle}/eval_files_{machine_type}_{machine_id}_{db}.pickle".format(
            root=dir_param["root_dir"],
            pickle=dir_param["pickle_dir"],
            machine_type=machine_type,
            machine_id=machine_id,
            db=db,
        )
    )

    eval_labels_pickle = (
        "{root}/{pickle}/eval_labels_{machine_type}_{machine_id}_{db}.pickle".format(
            root=dir_param["root_dir"],
            pickle=dir_param["pickle_dir"],
            machine_type=machine_type,
            machine_id=machine_id,
            db=db,
        )
    )

    model_file = "{root}/{model}/model_{machine_type}_{machine_id}_{db}.hdf5".format(
        root=dir_param["root_dir"],
        model=dir_param["model_dir"],
        machine_type=machine_type,
        machine_id=machine_id,
        db=db,
    )

    history_img = "{root}/{model}/history_{machine_type}_{machine_id}_{db}.png".format(
        root=dir_param["root_dir"],
        model=dir_param["model_dir"],
        machine_type=machine_type,
        machine_id=machine_id,
        db=db,
    )

    evaluation_result_key = "{machine_type}_{machine_id}_{db}".format(
        machine_type=machine_type, machine_id=machine_id, db=db
    )

# dataset generator
print("============== DATASET GENERATOR ==============")

if (
    os.path.exists(train_pickle)
    and os.path.exists(eval_files_pickle)
    and os.path.exists(eval_labels_pickle)
):
    train_data = pickler.load_pickle(train_pickle)
    eval_files = pickler.load_pickle(eval_files_pickle)
    eval_labels = pickler.load_pickle(eval_labels_pickle)
else:
    (
        train_files,
        train_labels,
        eval_files,
        eval_labels,
    ) = generate_data.dataset_generator(target_dir)

    train_data = generate_data.list_to_vector_array(
        train_files,
        msg="generate train_dataset",
        n_mels=model_param["feature"]["n_mels"],
        frames=model_param["feature"]["frames"],
        n_fft=model_param["feature"]["n_fft"],
        hop_length=model_param["feature"]["hop_length"],
        power=model_param["feature"]["power"],
    )

    pickler.save_pickle(train_pickle, train_data)
    pickler.save_pickle(eval_files_pickle, eval_files)
    pickler.save_pickle(eval_labels_pickle, eval_labels)

# model training #
model = keras_model(model_param["feature"]["n_mels"] * model_param["feature"]["frames"])
model.summary()

# training #
if os.path.exists(model_file):
    model.load_weights(model_file)
    print("============== MODEL ALREADY ==============")
else:
    model.compile(**model_param["fit"]["compile"])
    history = model.fit(
        train_data,
        train_data,
        epochs=model_param["fit"]["epochs"],
        batch_size=model_param["fit"]["batch_size"],
        shuffle=model_param["fit"]["shuffle"],
        validation_split=model_param["fit"]["validation_split"],
        verbose=model_param["fit"]["verbose"],
    )
    visualizer_train = visualizer()
    visualizer_train.loss_plot(history.history["loss"], history.history["val_loss"])
    visualizer_train.save_figure(history_img)
    visualizer_train.show()
    model.save_weights(model_file)

# evaluation
print("============== EVALUATION ==============")
y_pred = [0. for i in eval_labels]
y_true = eval_labels
for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
    try:
        data = generate_data.file_to_vector_array(file_name,
                                    n_mels=model_param["feature"]["n_mels"],
                                    frames=model_param["feature"]["frames"],
                                    n_fft=model_param["feature"]["n_fft"],
                                    hop_length=model_param["feature"]["hop_length"],
                                    power=model_param["feature"]["power"])
        error = numpy.mean(numpy.square(data - model.predict(data)), axis=1)
        y_pred[num] = numpy.mean(error)
    except:
        logger.warning("File broken!!: {}".format(file_name))
score = metrics.roc_auc_score(y_true, y_pred)
logger.info("AUC : {}".format(score))
evaluation_result["AUC"] = float(score)
results[evaluation_result_key] = evaluation_result
print("===========================")

# output results
print("\n===========================")
logger.info("all results -> {}".format(result_file))
with open(result_file, "w") as f:
    f.write(yaml.dump(results, default_flow_style=False))
print("===========================")
