/**
 * Middleware for processing queryes from main process to PPO algorithm.
 */
importScripts("https://cdnjs.cloudflare.com/ajax/libs/tensorflow/3.8.0/tf.min.js")
importScripts('../../neuralnetworks.js')
importScripts('../../utils.js')
importScripts("./PPO_class_image.js")

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
        self.postMessage({
            msg_type: "step", 
            action: self.env.action
        });
    });
}

function start(e){
        // console.log("PPOworker first onmessage");
        agent.observation_space = e.data.observation_space;
        agent.action_space = e.data.action_space;
        env.n_obs = e.data.n_obs;
        agent.imgshape = e.data.imgshape;
        let policy_nn = e.data.policy_nn;
        let obs_dim = agent.observation_space.shape;
        let act_dim = agent.action_space.shape
        let model;
        if (!policy_nn){
            model = create_cnn_model_by_serialized_data(policy_nn);
        } else {
            model = build_cnn(agent.imgshape, [64], act_dim[0], 'tanh', 'tanh');
        }
        self.ppo_obj = new PPO({env: env, agent: agent, hidden_sizes:[64], cr_lr:3e-4, ac_lr:3e-4, gamma:0.99, lam:0.95, steps_per_env:2000, 
            number_envs:1, eps:0.15, actor_iter:6, critic_iter:10, num_epochs:5000, minibatch_size:128, policy_nn: model, imgshape: agent.imgshape});
        self.ppo_obj.train();
}

async function getPolicyWeigts(){
    let pw = await self.ppo_obj.getPolicyWeights();
    let vw = await self.ppo_obj.getValueWeights();
    self.postMessage({
        msg_type: "get_policy_weights_answer", 
        policy_weights: pw,
        value_weights: vw
    });
}

async function loadPolicyModel(pw){
    let nn = create_model_by_serialized_data(pw);
    await self.ppo_obj.setPolicyModel(nn);
    self.postMessage({
        msg_type: "loaded_policy_weigths"
    });
}

async function loadValueModel(vw){
    let nn = create_model_by_serialized_data(vw);
    await self.ppo_obj.setValueModel(nn);
    self.postMessage({
        msg_type: "loaded_value_weights"
    });
}

//at first we get "agent" and "env"
tf.setBackend("webgl").then(()=>{
    self.onmessage = function(e){
        if (e.data.msg_type === "start"){
            start(e);
        }
        if (e.data.msg_type === "step"){
            self.env.e_l = e.data.e_l;
            self.env.e_r = e.data.e_r;
            self.resolve(e.data.step_data); 
        }
        if (e.data.msg_type === "get_policy_weights"){
            getPolicyWeigts();
        }
        if (e.data.msg_type === "set_weigths"){  
            if (e.data.policy_weights){
                loadPolicyModel(e.data.policy_weights)
            }
            if (e.data.value_weights){
                loadValueModel(e.data.value_weights)
            }
        }        
    }    
});
