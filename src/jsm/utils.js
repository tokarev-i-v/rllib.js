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

export function getRandomArbitrary(min, max) {
    return Math.random() * (max - min) + min;
}

export function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min)) + min;
  }
  