import * as tf from '@tensorflow/tfjs'
tf.disableDeprecationWarnings();

//import DQN from "./agents/DQN/BaseDQN";
import {PPOContinuous} from "./agents/policy_gradients/PPO";
//import FlatAreaEatWorld from "./envs/FlatAreaWorld/FlatAreaEatWorld_d"
import {FlatAreaEatWorld_c, Agent as FlatAgent} from "./envs/FlatAreaWorld/FlatAreaEatWorld_c"
import {TestWorld_c, Agent as TestAgent} from "./envs/TestWorld/TestWorld_c"
import {HuntersWorld, Agent as HunterAgent} from "./envs/HuntersWorld/HuntersWorld"
import {build_full_connected}  from './src/jsm/neuralnetworks';
import {SimpleUI} from './src/jsm/ui/SimplePPOUI'
let curretWorldClass = HuntersWorld;

var PPOworker = new Worker("agents/policy_gradients/ppo_worker.js");

let a = new HunterAgent({eyes_count: 10});
let cur_nn = build_full_connected(a.observation_space.shape, [128, 128], a.action_space.shape, 'tanh', 'tanh');
let ui = new SimpleUI({parent: document.body, policy_nn: cur_nn});
let w = new curretWorldClass({});
//const saveResult = cur_nn.save('downloads://mymodel');
w.addAgent(a);
PPOworker.onmessage = function(e){
    var step_data = w.step(e.data.action);
    PPOworker.postMessage({step_data: step_data, n_obs: w.n_obs, e_r: w.get_episode_reward(), e_l: w.get_episode_length()});
}


tf.setBackend("cpu").then(()=>{
    PPOworker.postMessage({
        observation_space: a.observation_space,
        action_space: a.action_space,
        n_obs: w.n_obs,
        policy_nn: cur_nn
    });
    
});

