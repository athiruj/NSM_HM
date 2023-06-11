
if os.path.exists(model_file):
    model.load_weights(model_file)
    print("============== MODEL ALREADY ==============")