import os
import yaml
import glob

import pickler
import generate_data

 # load parameter yaml
with open("./config/directory_config.yaml") as stream:
    dir_param = yaml.safe_load(stream)

with open("./config/model_config.yaml") as stream:
    model_param = yaml.safe_load(stream)

root_result_dir = dir_param["root_dir"]

 # make output directory to G drive
os.makedirs("{root}".format(root=root_result_dir), exist_ok=True)
os.makedirs("{root}{dir}".format(root=root_result_dir, dir=dir_param["pickle_dir"]), exist_ok=True)
os.makedirs("{root}{dir}".format(root=root_result_dir, dir=dir_param["pickle_dir"]), exist_ok=True)
os.makedirs("{root}{dir}".format(root=root_result_dir, dir=dir_param["model_dir"] ), exist_ok=True)
os.makedirs("{root}{dir}".format(root=root_result_dir, dir=dir_param["result_dir"]), exist_ok=True)

 # load base_directory list
dirs = sorted(
    glob.glob(
        os.path.abspath(".{dataset}/*/*/*".format(dataset=dir_param["dataset_dir"]))
    )
)

 # setup the result
result_file = "{result_dir}/{result_file}".format(result_dir=dir_param["result_dir"], result_file=dir_param["result_file"])
results = {}

 # loop of the dataset directory
for dir_idx, target_dir in enumerate(dirs):
    print("\n===========================")
    print("[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(dirs)))

    # dataset param        
    db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
    machine_type = os.path.split(os.path.split(target_dir)[0])[1]
    machine_id = os.path.split(target_dir)[1]

    # setup path
    evaluation_result = {}
    train_pickle = "{root}/{pickle}/train_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                pickle=dir_param["pickle_dir"],
                                                                                machine_type=machine_type,
                                                                                machine_id=machine_id, db=db
                                                                                )
    
    eval_files_pickle = "{root}/{pickle}/eval_files_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                   pickle=dir_param["pickle_dir"],
                                                                                   machine_type=machine_type,
                                                                                   machine_id=machine_id,
                                                                                   db=db
                                                                                   )
    
    eval_labels_pickle = "{root}/{pickle}/eval_labels_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                    pickle=dir_param["pickle_dir"],
                                                                                    machine_type=machine_type,
                                                                                    machine_id=machine_id,
                                                                                    db=db
                                                                                    )
    
    model_file = "{root}/{model}/model_{machine_type}_{machine_id}_{db}.hdf5".format(
                                                                            model=dir_param["model_dir"],
                                                                            machine_type=machine_type,
                                                                            machine_id=machine_id,
                                                                            db=db
                                                                            )
    
    history_img = "{model}/history_{machine_type}_{machine_id}_{db}.png".format(
                                                                                model=dir_param["model_dir"],
                                                                                machine_type=machine_type,
                                                                                machine_id=machine_id,
                                                                                db=db
                                                                                )
    
    evaluation_result_key = "{machine_type}_{machine_id}_{db}".format(
                                                                    machine_type=machine_type,
                                                                    machine_id=machine_id,
                                                                    db=db
                                                                    )
    
 # dataset generator
print("============== DATASET_GENERATOR ==============")

if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
    train_data = pickler.load_pickle(train_pickle)
    eval_files = pickler.load_pickle(eval_files_pickle)
    eval_labels = pickler.load_pickle(eval_labels_pickle)
else:
    train_files, train_labels, eval_files, eval_labels = generate_data.dataset_generator(target_dir)

    train_data = generate_data.list_to_vector_array(train_files,
                                              msg="generate train_dataset",
                                              n_mels=model_param["feature"]["n_mels"],
                                              frames=model_param["feature"]["frames"],
                                              n_fft=model_param["feature"]["n_fft"],
                                              hop_length=model_param["feature"]["hop_length"],
                                              power=model_param["feature"]["power"])

    pickler.save_pickle(train_pickle, train_data)
    pickler.save_pickle(eval_files_pickle, eval_files)
    pickler.save_pickle(eval_labels_pickle, eval_labels)

