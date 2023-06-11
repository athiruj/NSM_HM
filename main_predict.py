import os
import glob
import pickler
import yaml

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


predict_file = pickler.load_pickle(eval_files_pickle)
print(predict_file[0])
print(predict_file[-1])


data = generate_data.file_to_vector_array(
                predict_file[-1],
                n_mels=model_param["feature"]["n_mels"],
                frames=model_param["feature"]["frames"],
                n_fft=model_param["feature"]["n_fft"],
                hop_length=model_param["feature"]["hop_length"],
                power=model_param["feature"]["power"],
            )

model = keras_model(
            model_param["feature"]["n_mels"] * model_param["feature"]["frames"]
        )

for model_idx, model_file in enumerate(dirs):
    if os.path.exists(model_file):
        model.load_weights(model_file)
        print(
            "============== MODEL ALREADY [{idx}] ==============".format(idx=model_idx)
        )



prediction = model.predict(data[:1])
print("prediction shape:", prediction.shape)