function build_full_connected(input_shape, hiddens_config, output_shape, activation='relu', last_activation='relu'){
    return tf.tidy(()=>{
        if (typeof(input_shape) == "object"){
            input_shape = input_shape[0];
        }
        if (typeof(output_shape) == "object"){
            output_shape = output_shape[0];
        }
        let inputt = tf.input({shape: [null, input_shape]});
        let x = inputt;
        for(let l=0; l < hiddens_config.length; l++){
            x = tf.layers.dense({units:hiddens_config[l], activation:activation}).apply(x);
        }
        let output;
        if (last_activation){
            output = tf.layers.dense({units: output_shape, activation: last_activation}).apply(x);
        } else {
            output = tf.layers.dense({units: output_shape}).apply(x);
        }
        return tf.keep(tf.model({inputs:inputt, outputs:output}));
    });
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
        kernelSize: 3,
        filters: 8,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
        }));

    // x = tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}).apply(x);
    model.add(tf.layers.flatten());
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
        kernelSize: 3,
        filters: 8,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
        }));

    // x = tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}).apply(x);
    model.add(tf.layers.flatten());
    for(let l=0; l < hiddens_config.length; l++){
        model.add(tf.layers.dense({units:hiddens_config[l], activation:activation}));
    }
    model.add(tf.layers.dense({units: 1, activation: last_activation}));
    return model
}