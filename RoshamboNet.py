from tensorflow.keras import layers


def RoshamboNetPruningPolicy(pruning_policy):
    if pruning_policy["mode"] == "fixed":
        return pruning_policy["target"]
    else:
        raise AttributeError


def RoshamboNet(
        input_tensor,
        classes=4,
        include_top=True,
        pooling="max",
        num_3x3_blocks=3,
        **kwargs
):
    if pooling == "max":
        Pooling = layers.MaxPooling2D
    elif pooling == "avg":
        Pooling = layers.AveragePooling2D
    else:
        raise ValueError

    # Block 1
    num_out_channels = 16
    x = layers.Conv2D(num_out_channels, (5, 5),
                      activation='relu',
                      padding='valid',
                      name='layer1')(input_tensor)
    x = Pooling((2, 2), strides=(2, 2), name='pool1')(x)

    for block_idx in range(num_3x3_blocks):
        num_out_channels = num_out_channels * 2
        conv_name = "layer{}".format(block_idx + 2)
        pool_name = "pool{}".format(block_idx + 2)
        # Block 3x3
        x = layers.Conv2D(num_out_channels, (3, 3),
                          activation='relu',
                          padding='valid',
                          name=conv_name)(x)
        x = Pooling((2, 2), strides=(2, 2), name=pool_name)(x)

    conv_name = "layer{}".format(num_3x3_blocks + 2)
    pool_name = "pool{}".format(num_3x3_blocks + 2)
    x = layers.Conv2D(num_out_channels, (1, 1),
                      activation='relu',
                      padding='valid',
                      name=conv_name)(x)
    x = Pooling((2, 2), strides=(2, 2), name=pool_name)(x)

    # Block FC
    fc_name = "layer{}".format(num_3x3_blocks + 2 + 1)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(classes, name=fc_name, activation="softmax")(x)

    return x
