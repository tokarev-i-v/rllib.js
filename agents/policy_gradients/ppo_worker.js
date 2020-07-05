const { PPOContinuous } = require("./PPO")

const resolves = {}
const rejects = {}
let globalMsgId = 0;

// import {PPOContinuous} from PPO;

let agent = {
    observation_space: [],
    action_space: []
};
let env = {
    step: makeStep,
    n_obs: []
}

function makeStep(){
    return new Promise(function (resolve, reject) {
        postMessage();
        onmessage = function(e){
            env.n_obs = e.data.n_obs;
            resolve(e.data.step_data);   
        }
    });
  }

//at first we get "agent" and "env"
onmessage = function(e){
    agent.observation_space = e.data.observation_space;
    agent.action_space = e.data.action_space;
    console.log(e.data.observation_space);
    PPOContinuous({env: env, agent: agent, hidden_sizes:[64,64], cr_lr:5e-4, ac_lr:2e-4, gamma:0.99, lam:0.95, steps_per_env:10, 
        number_envs:10, eps:0.15, actor_iter:6, critic_iter:10, num_epochs:50, minibatch_size:64});
}

