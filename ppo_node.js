

import * as tf from '@tensorflow/tfjs-node-gpu';
import {Worker} from 'worker_threads';
import {JSDOM} from 'jsdom';
const jsdel =  new JSDOM(`<!DOCTYPE html><html><head></head><body>hello</body></html>`);
global.window = jsdel.window;
global.document = jsdel.window.document;
// tf.disableDeprecationWarnings();

import {HuntersWorld, Agent as HunterAgent} from "./envs/HuntersWorld/HuntersWorld_node.js";
import {build_full_connected}  from './src/jsm/neuralnetworks.js';
import {getWeightsFromModelToWorkerTransfer, setWeightsToModelByObject}  from './src/jsm/utils.js';
let curretWorldClass = HuntersWorld;

var PPOworker = new Worker("./agents/policy_gradients/ppo_class_worker.js");

var a = new HunterAgent({eyes_count: 10});
let cur_nn = build_full_connected(a.observation_space.shape, [128, 128], a.action_space.shape, 'tanh', 'tanh');
let weights_obj = getWeightsFromModelToWorkerTransfer(cur_nn);
// let ui = new SimpleUI({parent: document.body, policy_nn: cur_nn, worker: PPOworker});
var w = new curretWorldClass({});
//cur_nn = tf.loadLayersModel('http://localhost:1234/models/mymodel.json');
w.addAgent(a);
PPOworker.onmessage = function(e){
    if(e.data.msg_type === "step"){
        var step_data = w.step(e.data.action);
        PPOworker.postMessage({msg_type: "step", step_data: step_data, n_obs: w.n_obs, e_r: w.get_episode_reward(), e_l: w.get_episode_length()});
    }
    if(e.data.msg_type === "get_policy_weights"){
        let model = build_full_connected(a.observation_space.shape, [64,64], a.action_space.shape, 'tanh', 'tanh');
        model = setWeightsToModelByObject(model, e.data.policy_weights);
        model.save('downloads://mymodel');
    }
}


tf.setBackend("cpu").then(()=>{
    PPOworker.postMessage({
        msg_type: "start",
        observation_space: a.observation_space,
        action_space: a.action_space,
        n_obs: w.n_obs,
        policy_nn: weights_obj
    });
    
});
