import * as tf from '@tensorflow/tfjs-node';
// tf.enableDebugMode ()
import {
    act_smp_cont,
    get_p_noisy,
    get_p_log_cont,
    gaussian_log_likelihood,

    clipped_surrogate_obj, 
    discounted_rewards, 
    GAE, 
    Buffer,
    PPOContinuous} from "./PPO";

import {
    Agent,
    FlatAreaEatWorld_c} from "../../envs/FlatAreaWorld/FlatAreaEatWorld_c"


let a = new Agent({eyes_count: 10});
let s = a.sample_actions();
s.print();
let w = new FlatAreaEatWorld_c({});
w.addAgent(a);
PPOContinuous({env: w, agent: a, hidden_sizes:[64,64], cr_lr:5e-4, ac_lr:2e-4, gamma:0.99, lam:0.95, steps_per_env:5000, 
    number_envs:1, eps:0.15, actor_iter:6, critic_iter:10, num_epochs:5000, minibatch_size:256});
console.log(w.step([0,0,0]));

