import os
import glob

# import numpy
# from sklearn import metrics
# from tqdm import tqdm

import seaborn as sns
import pickler
import yaml
import matplotlib.pyplot as plt
import generate_data

from model import keras_model

with open("./config/directory_config.yaml") as stream:
    dir_param = yaml.safe_load(stream)

with open("./config/model_config.yaml") as stream:
    model_param = yaml.safe_load(stream)

dirs = sorted(
    glob.glob(
        os.path.abspath(
            "{root}/{model}/*.hdf5".format(
                root=dir_param["root_dir"],
                model=dir_param["model_dir"],
            )
        )
    )
)

eval_files_pickle = (
    "{root}/{pickle}/eval_files_{machine_type}_{machine_id}_{db}.pickle".format(
        root=dir_param["root_dir"],
        pickle=dir_param["pickle_dir"],
        machine_type="fan",
        machine_id="id_00",
        db="max_6db",
    )
)

eval_labels_pickle = (
    "{root}/{pickle}/eval_labels_{machine_type}_{machine_id}_{db}.pickle".format(
        root=dir_param["root_dir"],
        pickle=dir_param["pickle_dir"],
        machine_type="fan",
        machine_id="id_00",
        db="max_6db",
    )
)


predict_files = pickler.load_pickle(eval_files_pickle)
predict_labels = pickler.load_pickle(eval_labels_pickle)


data_n = generate_data.file_to_vector_array(
    predict_files[0],
    n_mels=model_param["feature"]["n_mels"],
    frames=model_param["feature"]["frames"],
    n_fft=model_param["feature"]["n_fft"],
    hop_length=model_param["feature"]["hop_length"],
    power=model_param["feature"]["power"],
)

# data_ab = generate_data.file_to_vector_array(
#     predict_files[-1],
#     n_mels=model_param["feature"]["n_mels"],
#     frames=model_param["feature"]["frames"],
#     n_fft=model_param["feature"]["n_fft"],
#     hop_length=model_param["feature"]["hop_length"],
#     power=model_param["feature"]["power"],
# )
# y_pred = [0.0 for k in predict_labels]

# print(y_pred)

model = keras_model(model_param["feature"]["n_mels"] * model_param["feature"]["frames"])

for model_idx, model_file in enumerate(dirs):
    if os.path.exists(model_file):
        model.load_weights(model_file)
        print(
            "============== MODEL ALREADY [{idx}] ==============".format(idx=model_idx)
        )


prediction_n = model.predict(data_n)
# prediction_ab = model.predict(data_ab)

# error_n = numpy.mean(numpy.square(data_n - prediction_n), axis=1)
# error_ab = numpy.mean(numpy.square(data_ab - prediction_ab), axis=1)
# n = numpy.mean(error_n)
# ab = numpy.mean(error_ab)
# print("prediction_n argmax:", prediction_n.argmax(), n)
# # print("prediction_n error_n:", error_n)
# print("prediction_ab argmax:", prediction_ab.argmax(), ab)
# # print("prediction_ab error_ab:", error_ab)


plt.show()

# # evaluation
# print("============== EVALUATION ==============")
# y_pred = [0.0 for k in predict_labels]
# y_true = predict_labels
# for num, file_name in tqdm(enumerate(predict_files), total=len(predict_files)):
#     try:
#         data = generate_data.file_to_vector_array(
#             file_name,
#             n_mels=model_param["feature"]["n_mels"],
#             frames=model_param["feature"]["frames"],
#             n_fft=model_param["feature"]["n_fft"],
#             hop_length=model_param["feature"]["hop_length"],
#             power=model_param["feature"]["power"],
#         )
#         error = numpy.mean(numpy.square(data - model.predict(data)), axis=1)
#         y_pred[num] = numpy.mean(error)
#     except:
#         # logger.warning("File broken!!: {}".format(file_name))
#         print("error")
# score = metrics.roc_auc_score(y_true, y_pred)
# # logger.info("AUC : {}".format(score))
# # evaluation_result["AUC"] = float(score)
# # results[evaluation_result_key] = evaluation_result
# print("===========================")

# print(y_true)
# print("===========================")
# print(y_pred)

# # output results
# c= 1
# print(y_true[c],y_pred[c])