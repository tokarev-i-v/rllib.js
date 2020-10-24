import { parentPort, MessagePort, workerData } from 'worker_threads';
const { PPO } = require("./PPO_class_node")
import {setWeightsToModelByObject}  from '../../src/jsm/utils_node';
import {build_full_connected}  from '../../src/jsm/neuralnetworks_node';

var self = this;
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

self.env = env;
function makeStep(){
    // console.log("start of makestep");
    return new Promise(function (resolve, reject) {
        // console.log("On send action, ", self.env.action);
        self.resolve = resolve;
        parentPort.postMessage({
            msg_type: "step", 
            action: self.env.action
        });
    });
}

function start(data){
        // console.log("PPOworker first onmessage");
        agent.observation_space = data.observation_space;
        agent.action_space = data.action_space;
        env.n_obs = data.n_obs;
        let policy_nn = data.policy_nn;
        let obs_dim = agent.observation_space.shape;
        let act_dim = agent.action_space.shape
        self.num_epochs = data.num_epochs ? data.num_epochs : 1000 
        let model = build_full_connected(obs_dim, [64,64], act_dim, 'tanh', 'tanh');
        if (policy_nn){
            model = setWeightsToModelByObject(model, policy_nn);
        }
        self.ppo_obj = new PPO({env: env, agent: agent, hidden_sizes:[64,64], cr_lr:5e-4, ac_lr:2e-4, gamma:0.99, lam:0.95, steps_per_env:1000, 
            number_envs:1, eps:0.15, actor_iter:6, critic_iter:10, num_epochs:self.num_epochs, minibatch_size:256, policy_nn: model});
        self.ppo_obj.train();
}

async function getPolicyWeigts(){
    let pw = await self.ppo_obj.getPolicyWeights();
    parentPort.postMessage({
        msg_type: "get_policy_weights", 
        policy_weights: pw
    });
}
parentPort.on('message', function(data){
    // console.log(data);
    if (data.msg_type === "start"){
        start(data);
    }
    if (data.msg_type === "step"){
        self.env.e_l = data.e_l;
        self.env.e_r = data.e_r;
        self.resolve(data.step_data); 
    }
    if (data.msg_type === "get_policy_weights"){
        getPolicyWeigts();
    }
});    
