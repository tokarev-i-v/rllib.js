import * as tf from "@tensorflow/tfjs";
import '@tensorflow/tfjs-node-gpu'
const { PPO } = require("./PPO_class")
import {setWeightsToModelByObject}  from '../../src/jsm/utils';
import {build_full_connected}  from '../../src/jsm/neuralnetworks';
var agent = {
    observation_space: [],
    action_space: []
};
var env = {
    step: makeStep,
    n_obs: [],
    action: null,
    e_r: 0,
    e_l: 0,
    reset: env_reset,
    get_episode_reward: get_episode_reward,
    get_episode_length: get_episode_length,
    
}

function get_episode_reward(){
    return env.e_r;
}
function get_episode_length(){
    return env.e_l;
}

function env_reset(){
    return 0;
}


this.env = env;
function makeStep(){
    // console.log("start of makestep");
    return new Promise(function (resolve, reject) {
        // console.log("On send action, ", this.env.action);
        this.resolve = resolve;
        this.postMessage({
            msg_type: "step", 
            action: this.env.action
        });
    });
}

function start(e){
        // console.log("PPOworker first onmessage");
        agent.observation_space = e.data.observation_space;
        agent.action_space = e.data.action_space;
        env.n_obs = e.data.n_obs;
        let policy_nn = e.data.policy_nn;
        let obs_dim = agent.observation_space.shape;
        let act_dim = agent.action_space.shape

        let model = build_full_connected(obs_dim, [64,64], act_dim, 'tanh', 'tanh');
        if (policy_nn){
            model = setWeightsToModelByObject(model, policy_nn);
        }
        this.ppo_obj = new PPO({env: env, agent: agent, hidden_sizes:[64,64], cr_lr:5e-4, ac_lr:2e-4, gamma:0.99, lam:0.95, steps_per_env:1000, 
            number_envs:1, eps:0.15, actor_iter:6, critic_iter:10, num_epochs:5000, minibatch_size:256, policy_nn: model});
        this.ppo_obj.train();
}

async function getPolicyWeigts(){
    let pw = await this.ppo_obj.getPolicyWeights();
    this.postMessage({
        msg_type: "get_policy_weights", 
        policy_weights: pw
    });
}

//at first we get "agent" and "env"
// tf.setBackend("cpu").then(()=>{
    this.onmessage = function(e){
        if (e.data.msg_type === "start"){
            start(e);
        }
        if (e.data.msg_type === "step"){
            this.env.e_l = e.data.e_l;
            this.env.e_r = e.data.e_r;
            this.resolve(e.data.step_data); 
        }
        if (e.data.msg_type === "get_policy_weights"){
            getPolicyWeigts();
        }
    }    
// });
