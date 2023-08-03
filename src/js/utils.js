/**
 * Setting parameters as default config.
 * object's members names will be used for
 * @param {*} default_params object with default_parameters
 * @param {*} params 
 */
function params_setter(default_params, params){
    for(let param_name in default_params){
        try{
            if(typeof(params[param_name]) == typeof(default_params[param_name])){
              this[param_name] = params[param_name];
            } else {
              this[param_name] = default_params[param_name];
            }
      
        }catch(e){
            console.log(`CHECK ${param_name} PARAMETER`);
            this[param_name] = default_params[param_name];
        }
    
    }
}

/**
 * 
 * @param {*} min 
 * @param {*} max 
 */
function getRandomArbitrary(min, max) {
    return Math.random() * (max - min) + min;
}

/**
 * 
 * @param {*} min 
 * @param {*} max 
 */
function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min)) + min;
  }

/**
 * 
 * @param {*} model 
 */
function getWeightsFromModelToWorkerTransfer(model){
    let ret = {};
    if(model){
        for(let layer in model.layers){
            ret[layer] = {};
            for(let wg of model.layers[layer].getWeights()){
                ret[layer][wg.name] = wg.arraySync();

            }    
        }
    }
    return ret;
}

/**
 * 
 * @param {tf.model} model 
 * @param {object} weights_obj 
 */
function setWeightsToModelByObject(model, weights_obj){
    if(model){
        for(let layer in model.layers){
            for(let wg of model.layers[layer].getWeights()){
                if(weights_obj[layer][wg.name]){
                    let new_weights = tf.tensor(weights_obj[layer][wg.name]);
                    model.layers[layer][wg.name] = new_weights;
                }    
            }    
        }
    }
    return model;
}




/**
 * 
 * @param {tf.model} model  Tensorflow.js LayersModel
 * 
 *  let layerData = [
        {
            "name": "dense_Dense3",
            "layers": {
                "dense_Dense3/bias": {
                    shape: [10],
                    layer_data: []
                },
                "dense_Dense3/kernel":  {
                    shape: [10],
                    layer_data: []
                }
            }
        },
        {
            "name": "dense_Dense4",
            "layers": {
                "dense_Dense4/bias": {
                    shape: [10],
                    layer_data: []
                },
                "dense_Dense4/kernel":  {
                    shape: [10],
                    layer_data: []
                }
            }
        }        
    ];
 */
function get_serialized_layers_data(model){

    if(model){
        let layersData = [];
        for(let layer of model.layers){
            let layer_config = layer.getConfig();
            let layer_name = layer.name;
            let layer_shape = null;
            let layer_activation = null;
            if (layer_name.substring(0, 5) == "input"){
                layer_shape = layer.inputSpec[0].shape;
                if (layer_shape.length > 2){
                    layer_shape = layer_shape.slice(1, layer_shape.length);
                }
            } else {
                layer_shape = layer.units;
                layer_activation = layer_config.activation;
            }
            let layer_weights = [];
            for (let ld of layer.getWeights()){
                let weight = ld.arraySync();
                layer_weights.push(weight);
            }

            let layerDataItem = {
                "name": layer_name,
                "shape": layer_shape,
                "layer_weights": layer_weights,
                "activation": layer_activation
            }
            layersData.push(layerDataItem);
        }
        return layersData;
    }
    throw Error("Model must be specified.")
}

function create_model_by_serialized_data(model_weight_data){
    if(model_weight_data){
        let inputt = tf.input({shape: model_weight_data[0].shape});
        let cur_layer = inputt;
        for(let layer of model_weight_data.slice(1, model_weight_data.length)){
            let layer_name = layer.name;
            let layer_shape = layer.shape;
            let layer_activation = layer.activation;
            cur_layer = tf.layers.dense({units: layer_shape, activation: layer_activation}).apply(cur_layer);
        }
        let model = tf.model({inputs: inputt, outputs: cur_layer});
        for (let layer_number in model_weight_data){
            let layer_weights = model_weight_data[layer_number].layer_weights;
            if (layer_weights && layer_weights.length > 0){
                for (let i in layer_weights){
                    layer_weights[i] = tf.tensor(layer_weights[i]);
                }
                model.layers[layer_number].setWeights(model_weight_data[layer_number].layer_weights);
            }
        }
        return model;
    }
    throw Error("Model must be specified.")
}