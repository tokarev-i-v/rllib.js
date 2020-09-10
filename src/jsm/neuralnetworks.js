import * as tf from '@tensorflow/tfjs'

export function build_full_connected(input_shape, hiddens_config, output_shape, activation='relu', last_activation='relu'){
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
        let output = tf.layers.dense({units: output_shape, activation: last_activation}).apply(x);
        return tf.keep(tf.model({inputs:inputt, outputs:output}));
    });
}
