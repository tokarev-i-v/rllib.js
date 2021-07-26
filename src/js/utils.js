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
    let layerData = [
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

    if(model){
        let layersData = [];
        for(let layer of model.layers){
            let layer_name = layer.name;
            let layer_shape = null;
            if (layer_name.substring(0, 6) == "input"){
                layer_shape = layer.inputSpec[0].shape;
                if (layer_shape.length > 2){
                    layer_shape = layer_shape.slice(1);
                }
            } else {
                layer_shape = [layer.units];
            }
            let layer_weights = [];
            for (let ld of layer.getWeights()){
                let weight = ld.arraySync();
                layer_weights.push(weight);
            }

            let layerDataItem = {
                "name": layer_name,
                "shape": layer_shape,
                "layer_weights": layer_weights
            }
            layersData.push(layerDataItem);
        }
        return layersData;
    }
    throw Error("Model must be specified.")
}