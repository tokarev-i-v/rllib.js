import * as tf from '@tensorflow/tfjs'
//import DQN from "./agents/DQN/BaseDQN";
import {PPOContinuous} from "./agents/policy_gradients/PPO";
//import FlatAreaEatWorld from "./envs/FlatAreaWorld/FlatAreaEatWorld_d"
import {FlatAreaEatWorld_c, Agent} from "./envs/FlatAreaWorld/FlatAreaEatWorld_c"



// let world = new FlatAreaEatWorld({agent: DQN});
var PPOworker = new Worker("agents/policy_gradients/ppo_worker.js");

let a = new Agent({eyes_count: 10});
let w = new FlatAreaEatWorld_c({});
w.addAgent(a);
tf.disableDeprecationWarnings();
PPOworker.postMessage({
    observation_space: a.observation_space,
    action_space: a.action_space
});
// tf.setBackend("cpu").then(()=>{
//     PPOContinuous({env: w, agent: a, hidden_sizes:[64,64], cr_lr:5e-4, ac_lr:2e-4, gamma:0.99, lam:0.95, steps_per_env:10, 
//         number_envs:1, eps:0.15, actor_iter:6, critic_iter:10, num_epochs:50, minibatch_size:64});

// });

