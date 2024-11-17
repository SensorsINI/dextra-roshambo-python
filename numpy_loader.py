import os


def load_from_numpy(model, path):
    import glob
    import numpy as np
    log.info(f'loading {model} from {path}')
    shifts_weights = np.load(os.path.join(path, "shift_per_layer_kernel.npy"))
    shifts_biases = np.load(os.path.join(path,"shift_per_layer_bias.npy"))
    shifts_outputs = np.load(os.path.join(path,"shift_per_layer_activation.npy"))

    layer_index = 0

    for layer in model.layers:
        print(f'loading weights for layer {layer.name}')
        name = layer.name.replace("quant_", "")

        file_weights = glob.glob(path + r"/" + name + "*kernel.npy")
        file_bias = glob.glob(path + r"/" + name + "*bias.npy")

        if len(file_weights) == 0 and len(file_bias) == 0:
            print("***No weight found for layer {}***".format(name))
        else:
            print("Loading quantized weights for layer {}".format(name))

            if len(file_weights) != 1 or len(file_bias) != 1:
                raise FileNotFoundError(f'weights or biases npy file(s) missing for layer {layer.name}')

            weights = np.load(file_weights[0]).astype(np.float32) / (2.0 ** float(shifts_weights[layer_index]))
            biases = np.load(file_bias[0]).astype(np.float32) / (2.0 ** float(shifts_biases[layer_index]))

            current_weights = layer.get_weights()

            load_list = [weights, biases]


            model.get_layer(layer.name).set_weights(load_list)

            comparison = weights == model.get_layer(layer.name).get_weights()[0]
            equal_arrays = comparison.all()
            if bool(equal_arrays) is not True:
                raise ValueError(f'layer size does not match stored weight size for layer {layer.name}')

            comparison = biases == model.get_layer(layer.name).get_weights()[1]
            equal_arrays = comparison.all()
            if bool(equal_arrays) is not True:
                raise ValueError(f'model biases size does not match npy biases for layer {layer.name}')

            layer_index = layer_index + 1
    print("Done loading weights")
    return model
