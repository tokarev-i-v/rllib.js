import * as tf from '@tensorflow/tfjs-node-gpu'

/**
 * Setting parameters as default config.
 * object's members names will be used for
 * @param {*} default_params object with default_parameters
 * @param {*} params 
 */
export function params_setter(default_params, params){
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
export function getRandomArbitrary(min, max) {
    return Math.random() * (max - min) + min;
}

/**
 * 
 * @param {*} min 
 * @param {*} max 
 */
export function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min)) + min;
  }

/**
 * 
 * @param {*} model 
 */
export function getWeightsFromModelToWorkerTransfer(model){
    let ret = {};
    if(model){
        for(let layer in model.layers){
            // ret[layer] = [];
            ret[layer] = {};
            for(let wg of model.layers[layer].getWeights()){
                let obj_to_add = {};
                // obj_to_add[wg.name] = wg.arraySync();
                // ret[layer].push(obj_to_add);
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
export function setWeightsToModelByObject(model, weights_obj){
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