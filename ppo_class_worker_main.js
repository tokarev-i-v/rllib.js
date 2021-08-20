import * as tf from '@tensorflow/tfjs'
tf.disableDeprecationWarnings();

import {FlatAreaEatWorld_c, Agent as FlatAgent} from "./src/jsm/envs/FlatAreaWorld/FlatAreaEatWorld_c"
import {TestWorld_c, Agent as TestAgent} from "./src/jsm/envs/TestWorld/TestWorld_c"
import {HuntersWorld, Agent as HunterAgent} from "./src/jsm/envs/HuntersWorld/HuntersWorld"
import {HuntersWorld as HuntersWorld3D, Agent as HunterAgent3D} from "./src/jsm/envs/World3D/HuntersWorld3D"
import {build_full_connected}  from './src/jsm/neuralnetworks';
import {get_serialized_layers_data, create_model_by_serialized_data}  from './src/jsm/utils';
import {SimpleUI} from './src/jsm/ui/SimplePPOUI'
let curretWorldClass = HuntersWorld3D;

var PPOworker = new Worker("./src/jsm/agents/policy_gradients/ppo_class_worker.js");

var a = new HunterAgent3D({eyes_count: 10});
let cur_nn = build_full_connected(a.observation_space.shape, [128, 128], a.action_space.shape, 'tanh', 'tanh');
/*Create neural network*/
let weights_obj = get_serialized_layers_data(cur_nn);

/*Create UI*/
let ui = new SimpleUI({parent: document.body, policy_nn: cur_nn, worker: PPOworker});

/*Create environment*/
var w = new curretWorldClass({});

/*Adding agent to the environment object*/
w.addAgent(a);

PPOworker.onmessage = async function(e){
    if(e.data.msg_type === "step"){
        var step_data = w.step(e.data.action);
        PPOworker.postMessage({msg_type: "step", step_data: step_data, n_obs: w.n_obs, e_r: w.get_episode_reward(), e_l: w.get_episode_length()});
    }
    /*When user recieves neural network weights from agent worker*/
    if(e.data.msg_type === "get_policy_weights_answer"){
        /*create neural network from serialized weights*/
        let model_p = create_model_by_serialized_data(e.data.policy_weights);
        /*downloading policy weights*/
        model_p.save('downloads://policy');

        /*Create neural network from serialized weights*/
        let model_v = create_model_by_serialized_data(e.data.value_weights);
        /*downloading value function weights*/
        model_v.save('downloads://value');

    }
    /*Answer from worker that policy weights have been set*/
    if(e.data.msg_type === "load_policy_weigths_by_path_answer"){
        alert('Policy weights have been set');
    }
    /*Answer from worker that value function weights have been set*/
    if(e.data.msg_type === "load_value_weigths_by_path_answer"){
        alert('Value weights have been set');
    }
}


tf.setBackend("webgl").then(()=>{
    PPOworker.postMessage({
        msg_type: "start",
        observation_space: a.observation_space,
        action_space: a.action_space,
        n_obs: w.n_obs,
        policy_nn: weights_obj
    });
});

