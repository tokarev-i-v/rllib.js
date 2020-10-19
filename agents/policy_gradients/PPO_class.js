// import * as tf from '@tensorflow/tfjs-node-gpu';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node-gpu'
import {build_full_connected}  from '../../src/jsm/neuralnetworks';
import {getWeightsFromModelToWorkerTransfer}  from '../../src/jsm/utils';
const softmax_entropy = (logits) => tf.tidy(()=>{
    return tf.keep(tf.sum(tf.softmax(logits, dim=-1).mul(tf.logSoftmax(logits, axis=-1)), -1))
});
/**
 * PASSED
 * @param {tf.Tensor} new_p 
 * @param {tf.Tensor} old_p 
 * @param {tf.Tensor} adv 
 * @param {tf.Tensor} eps 
 */
export const clipped_surrogate_obj = (new_p, old_p, adv, eps) => tf.tidy(()=>{
    if (!(new_p instanceof tf.Tensor)){
        new_p = tf.tensor(new_p);
    }
    if (!(old_p instanceof tf.Tensor)){
        old_p = tf.tensor(old_p);
    }
    if (eps instanceof tf.Tensor){
        eps = eps.dataSync()[0];
    }
    let rt = new_p.sub(old_p).exp().flatten();
    let rtmul = rt.mul(adv);
    let clipped = tf.clipByValue(rt, 1-eps, 1+eps).mul(adv);
    let minimum = tf.minimum(rtmul, clipped);
    let minus_meaned = minimum.mean().mul(-1);
    return minus_meaned;
});

/**
 * PASSED
 * @param {tf.Tensor} rews 
 * @param {number} last_sv 
 * @param {number} gamma 
 */
export const discounted_rewards = (rews, last_sv, gamma) => tf.tidy(()=>{
    if(typeof(gamma) !== "number"){
        gamma = gamma.toFloat()
    }
    let rtg = rews.bufferSync();
    rtg.set(rtg.get(rews.shape[0]-1) + gamma * last_sv, rews.shape[0]-1);
    for(let i = rews.shape[0]-2;i >= 0; i--){
        rtg.set(rtg.get(i) + gamma * rtg.get(i+1), i);
    }

    let ret_value = tf.keep(rtg.toTensor());
    return ret_value;
});
/**
 * PASSED
 * @param {tf.Tensor} rews 
 * @param {tf.Tensor} v 
 * @param {Number} v_last 
 * @param {Number} gamma 
 * @param {Number} lam 
 */
export const GAE = (rews, v, v_last, gamma=0.99, lam=0.95) => tf.tidy(()=>{

    let vs = v.concat(tf.tensor([v_last]));
    gamma = tf.scalar(gamma);
    let delta = rews.add(vs.slice([1], [vs.shape[0]-1]).mul(gamma)).sub(vs.slice([0], [vs.shape[0]-1]));

    let gae_advantage = discounted_rewards(delta, 0, gamma.dataSync()*lam)

    return gae_advantage;
});

/**
 * PASSED
 * Contains replaybuffer for PPO.
 */
class Buffer{

    constructor(gamma=0.99, lam=0.95){
        this.gamma = gamma;
        this.lam = lam;
        this.adv = tf.tensor([]);
        this.ob = tf.tensor([]);
        this.ac = tf.tensor([]);
        this.rtg = tf.tensor([]);
    }
    store(temp_states, temp_rewards, temp_actions, temp_values, last_sv){
        if (temp_states.length > 0){
            tf.tidy(()=>{
                let t_s = tf.tensor(temp_states);
                let t_r = tf.tensor(temp_rewards);
                let t_a = tf.tensor(temp_actions);
                let t_v = tf.tensor(temp_values);
                this.ob = tf.keep(this.ob.concat(t_s));
                let rtg = discounted_rewards(t_r.flatten(), last_sv, this.gamma);
                this.adv = tf.keep(this.adv.concat(GAE(t_r.flatten(), t_v.flatten(), last_sv, this.gamma, this.lam)));
                this.rtg = tf.keep(this.rtg.concat(rtg));
                this.ac = tf.keep(this.ac.concat(t_a));    
            });
        }
    }
    get_batch(){
        return tf.tidy(()=> {
            let norm_adv = tf.keep(this.adv.sub(this.adv.mean()).div(tf.moments(this.adv).variance.sqrt().add(tf.scalar(1e-10))));
            return [tf.keep(this.ob), tf.keep(this.ac), norm_adv, tf.keep(this.rtg)];    
        });
    }
    len(){
        return this.ob.shape[0];
    }
}


/**
 * PASSED
 * Contains replaybuffer for PPO.
 */
export class Buffer_a{

    constructor(gamma=0.99, lam=0.95){
        this.gamma = gamma;
        this.lam = lam;
        this.adv = [];
        this.ob = [];
        this.ac = [];
        this.rtg = [];
    }
    store(temp_states, temp_rewards, temp_actions, temp_values, last_sv){
        if (temp_states.length > 0){
            tf.tidy(()=>{
                let t_s = tf.tensor(temp_states);
                let t_r = tf.tensor(temp_rewards);
                let t_a = tf.tensor(temp_actions);
                let t_v = tf.tensor(temp_values);
                // console.log("store 0", temp_traj[0]);
                this.ob = this.ob.concat(t_s.arraySync());
                // this.ob = tf.keep(this.ob.concat(tens.slice([0], [temp_traj.length,1]).flatten()));
                // console.log("store 1");
                // this.ob.print();
                let rtg = discounted_rewards(t_r.flatten(), last_sv, this.gamma);
                // console.log("store 2");
                this.adv = this.adv.concat(GAE(t_r.flatten(), t_v.flatten(), last_sv, this.gamma, this.lam).arraySync());
                // console.log("store 3");
                // this.adv.print();
                // console.log(this.adv.shape);
                this.rtg = this.rtg.concat(rtg.arraySync());
                // console.log("store 4");
                // this.rtg.print();
                this.ac = this.ac.concat(t_a.arraySync());    
                // console.log("store 5");
                // console.log(this.ob.shape);
                // console.log(this.adv.shape);
                // console.log(this.rtg.shape);
                // console.log(this.ac.shape);
            });
        }
    }
    get_batch(){
        return tf.tidy(()=> {
            let adv = tf.keep(tf.tensor(this.adv));
            let ob = tf.keep(tf.tensor(this.ob));
            let ac = tf.tensor(this.ac);
            let rtg = tf.keep(tf.tensor(this.rtg));
            
            let norm_adv = tf.keep(adv.sub(adv.mean()).div(tf.moments(adv).variance.sqrt().add(tf.scalar(1e-10))));
            return [ob, ac, norm_adv, rtg];    
        });
    }
    len(){
        return this.ob.length;
    }
}


/**
 * 
 * @param {tf.Tensor} policy_nn 
 */
export function act_smp_discrete(policy_nn){
    return tf.tidy(()=>{tf.squeeze(tf.multinomial(policy_nn, 1))});
}

/**
 * PASSED!
 * @param {tf.Tensor} p_noisy 
 * @param {Number} low_action_space 
 * @param {Number} high_action_space 
 */
export function act_smp_cont(p_noisy, low_action_space, high_action_space){
    return tf.tidy(()=>{
        return tf.keep(tf.clipByValue(p_noisy, low_action_space, high_action_space));
    });
}
/**
 * PASSED!
 * @param {tf.Tensor} policy_nn 
 * @param {tf.Tensor} log_std 
 */
export function get_p_noisy(policy_nn, log_std){
    return tf.tidy(()=>{
        return tf.keep(policy_nn.add(tf.randomNormal(policy_nn.shape, 0, 1).mul( tf.exp(log_std))));
    });
}

/**
 * 
 * @param {tf.Tensor} policy_nn 
 * @param {tf.Tensor} act_ph 
 * @param {Number} act_dim 
 * @param {tf.Tensor} log_std 
 */
export function get_p_log_discrete(policy_nn, act_ph, act_dim, log_std){
    return tf.tidy(()=>{
        let act_onehot = tf.oneHot(act_ph, depth=act_dim);
        return tf.sum(act_onehot.mul(tf.logSoftmax(policy_nn), axis=-1));    
    })
}

/**
 * PASSED!
 * @param {tf.Tensor} x 
 * @param {tf.Tensor} mean 
 * @param {tf.Tensor} log_std 
 */
export function get_p_log_cont(x, mean, log_std){
    return gaussian_log_likelihood(x, mean, log_std)
}

/**
 * PASSED!
 * @param {tf.Tensor} x 
 * @param {tf.Tensor} mean 
 * @param {tf.Tensor} log_std 
 */
export function gaussian_log_likelihood(x, mean, log_std){
    return tf.tidy(()=>{
        let divider = log_std.exp().square().add(tf.scalar(1e-9));
        let log_p = x.sub(mean).square().div(divider).add(log_std.mul(tf.scalar(2))).add(tf.scalar(2*Math.PI).log()).mul(tf.scalar(-0.5))
        return tf.keep(tf.sum(log_p, -1));    
    });
}

export class PPO{
    constructor(opt){

        let env = opt.env;
        let agent = opt.agent;
        this.hidden_sizes=opt.hidden_sizes; 
        this.cr_lr=opt.cr_lr; 
        this.ac_lr=opt.ac_lr;
        this.num_epochs=opt.num_epochs;
        this.minibatch_size=opt.minibatch_size; 
        this.gamma=opt.gamma;
        this.lam=opt.lam;
        this.number_envs=opt.number_envs; 
        this.eps=opt.eps;
        this.actor_iter=opt.actor_iter;
        this.critic_iter=opt.critic_iter;
        this.steps_per_env=opt.steps_per_env;
    
        this.envs = [];
        this.envs.push(env);
        this.agents = [];
        this.agents.push(agent);
        this.obs_dim = this.agents[0].observation_space.shape;
    
        this.low_action_space = this.agents[0].action_space.low
        this.high_action_space = this.agents[0].action_space.high
        this.act_dim = this.agents[0].action_space.shape

        // this.policy_nn = build_full_connected(this.obs_dim, this.hidden_sizes, this.act_dim, 'tanh', 'tanh');
        if (opt.policy_nn) { 
            this.policy_nn = opt.policy_nn; 
        } 
        this.log_std = tf.variable(tf.fill(this.act_dim, -0.5), false, 'log_std');
        
        this.p_noisy = get_p_noisy;
        this.act_smp = act_smp_cont;
        this.p_log = get_p_log_cont;
        this.s_values = build_full_connected(this.obs_dim, this.hidden_sizes, [1], 'tanh', null);
        this.p_opt = tf.train.adam(this.ac_lr);
        this.v_opt = tf.train.adam(this.cr_lr);
    }

    async getPolicyWeights(){
       let wghts = getWeightsFromModelToWorkerTransfer(this.policy_nn);
       return wghts;
    }

    async train(){

        let step_count = 0;
        for(let ep=0; ep<this.num_epochs;ep++){
            let buffer = new Buffer_a(this.gamma, this.lam);
            let batch_rew = [];
            let batch_len = [];        
            for(let env of this.envs){
                let temp_states = [];
                let temp_rewards = [];
                let temp_actions = [];
                let temp_values = [];
                for(let i=0; i < this.steps_per_env; i++){
                    // console.log(i);
                    let nobs = tf.tensor([env.n_obs]);
                    let nobs_e = tf.expandDims(nobs, 0);
                    let policy_nn_val = this.policy_nn.apply(nobs);
                    let p_noisy_val = this.p_noisy(policy_nn_val, this.log_std);
                    let act1 = this.act_smp(p_noisy_val, this.low_action_space, this.high_action_space);
                    let val = this.s_values.apply(nobs);
                    let act = tf.squeeze(act1);
                    env.action = act.arraySync();
                    let [obs2, rew, done, _] = await env.step();
                    // console.log("after await");
                    temp_states.push([env.n_obs.slice()])
                    temp_rewards.push([rew]);
                    temp_actions.push([act.arraySync()]);
                    let squeezed_val = tf.squeeze(val);
                    temp_values.push([squeezed_val.arraySync()]);
                    env.n_obs = obs2.slice();
                    step_count += 1;
                    if (done){
                        buffer.store(temp_states, temp_rewards, temp_actions, temp_values, 0);
                        temp_states = [];
                        temp_rewards = [];
                        temp_actions = [];
                        temp_values = [];
                        batch_rew.push(env.get_episode_reward())
                        batch_len.push(env.get_episode_length())
                        
                        env.reset()           
                    }
                    nobs.dispose();
                    policy_nn_val.dispose();
                    nobs_e.dispose();
                    p_noisy_val.dispose();
                    val.dispose();
                    squeezed_val.dispose();
                    act1.dispose();
                    act.dispose();
                }

                let nobs = tf.tensor([env.n_obs]);
                nobs = tf.expandDims(nobs, 0);
                let last_v = this.s_values.apply(nobs);
                buffer.store(temp_states, temp_rewards, temp_actions, temp_values, last_v);
            }        

        let [obs_batch, act_batch, adv_batch, rtg_batch] = buffer.get_batch();
        let policy_nn_value = this.policy_nn.apply(obs_batch);
        let old_p_log = this.p_log(act_batch, policy_nn_value, this.log_std);
        let old_p_batch = old_p_log;

        let lb = buffer.len();
        let shuffled_batch = tf.util.createShuffledIndices(lb); 

        for(let j=0; j < this.actor_iter; j++){
            tf.util.shuffle(shuffled_batch);
            for(let idx = 0; idx < lb; idx += this.minibatch_size){
                let minib = shuffled_batch.slice(idx, Math.min(idx+this.minibatch_size,lb));
                // let old_p_log_ph = old_p_batch[minib];
                let gat_tensor = tf.tensor(new Int32Array(minib));
                let p_loss = ()=> tf.tidy(()=>{
                    let policy_nn_value = this.policy_nn.apply(obs_batch.gather(gat_tensor));
                    let p_log_value = this.p_log( act_batch.gather(gat_tensor), policy_nn_value, this.log_std)
                    let p_loss_v = clipped_surrogate_obj(p_log_value, old_p_batch.gather(gat_tensor), adv_batch.gather(gat_tensor), this.eps)
                    return p_loss_v;    
                });
                
                let gradients = tf.variableGrads(p_loss, this.policy_nn.getWeights());
                this.p_opt.applyGradients(gradients.grads);
                tf.dispose(gradients);
                tf.dispose(gat_tensor);
            }

        }

        for(let j=0; j < this.critic_iter; j++){
            tf.util.shuffle(shuffled_batch);
            for(let idx = 0; idx < lb; idx += this.minibatch_size){
                let minib = shuffled_batch.slice(idx, Math.min(idx+this.minibatch_size,lb));
                let gat_tensor = tf.tensor(new Int32Array(minib) );
                let v_loss = ()=> tf.tidy(()=>{
                    let s_v = tf.squeeze(this.s_values.apply(obs_batch.gather(gat_tensor)));
                    let v_loss_v = tf.mean(rtg_batch.gather(gat_tensor).sub(s_v).square());
                    return v_loss_v;
                });

                let  gradients = tf.variableGrads(v_loss, this.s_values.getWeights());
                this.v_opt.applyGradients(gradients.grads);
                tf.dispose(gradients);
                tf.dispose(gat_tensor);
            }    
            
        }
        obs_batch.dispose();
        act_batch.dispose();
        adv_batch.dispose();
        rtg_batch.dispose();

        }


    }

    async eval(){

    }
}