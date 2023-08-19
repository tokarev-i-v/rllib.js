function build_full_connected(input_shape, hiddens_config, num_outputs, activation='relu', last_activation='relu'){
    let model = tf.sequential();
    model.add(tf.layers.dense({inputShape: input_shape, units:hiddens_config[0], activation:activation}));
    for(let l=1; l < hiddens_config.length; l++){
        model.add(tf.layers.dense({units:hiddens_config[l], activation:activation}));
    }
    if (last_activation){
        model.add(tf.layers.dense({units: num_outputs, activation: last_activation}));
    } else {
        model.add(tf.layers.dense({units: num_outputs}));
    }
    return model;
}

/** Build CNN network for image recognition tasks.
 * 
 * @param {Array} input_shape Input shape [img_width, img_height, channels] 
 * @param {Number} hiddens Count of hidden layers
 * @param {Number} output_shape Count of output neurons 
 * @param {String} activation Name of activation function 
 * @param {String} last_activation Name of last activation function 
 */
function build_cnn(input_shape, hiddens_config, output_shape, activation='relu', last_activation='relu'){
    let model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [input_shape[0], input_shape[1], input_shape[2]],
        kernelSize: 2,
        filters: 4,
        activation: 'relu'
    }));
    model.add(tf.layers.conv2d({
        kernelSize: 8,
        filters: 4,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tf.layers.flatten({activation:'relu'}));
    for(let l=0; l < hiddens_config.length; l++){
        model.add(tf.layers.dense({units:hiddens_config[l], activation:activation}));
    }
    model.add(tf.layers.dense({units: output_shape, activation: last_activation}));
    return model
}

function build_cnn_value(input_shape, hiddens_config, activation='relu', last_activation='relu'){
    let model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [input_shape[0], input_shape[1], input_shape[2]],
        kernelSize: 2,
        filters: 4,
        activation: 'relu'
    }));
    model.add(tf.layers.conv2d({
        kernelSize: 8,
        filters: 4,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tf.layers.flatten({activation:'relu'}));
    for(let l=0; l < hiddens_config.length; l++){
        model.add(tf.layers.dense({units:hiddens_config[l], activation:activation}));
    }
    model.add(tf.layers.dense({units: 1, activation: last_activation}));
    return model
}