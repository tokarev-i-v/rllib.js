// import * as tf from '@tensorflow/tfjs-node-gpu';
import * as tf from '@tensorflow/tfjs';

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
export class Buffer{

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
                // console.log("store 0", temp_traj[0]);
                this.ob = tf.keep(this.ob.concat(t_s));
                // this.ob = tf.keep(this.ob.concat(tens.slice([0], [temp_traj.length,1]).flatten()));
                // console.log("store 1");
                // this.ob.print();
                let rtg = discounted_rewards(t_r.flatten(), last_sv, this.gamma);
                // console.log("store 2");
                this.adv = tf.keep(this.adv.concat(GAE(t_r.flatten(), t_v.flatten(), last_sv, this.gamma, this.lam)));
                // console.log("store 3");
                // this.adv.print();
                // console.log(this.adv.shape);
                this.rtg = tf.keep(this.rtg.concat(rtg));
                // console.log("store 4");
                // this.rtg.print();
                this.ac = tf.keep(this.ac.concat(t_a));    
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
 * @param {tf.Tensor} p_logits 
 */
export function act_smp_discrete(p_logits){
    return tf.tidy(()=>{tf.squeeze(tf.multinomial(p_logits, 1))});
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
 * @param {tf.Tensor} p_logits 
 * @param {tf.Tensor} log_std 
 */
export function get_p_noisy(p_logits, log_std){
    return tf.tidy(()=>{
        return tf.keep(p_logits.add(tf.randomNormal(p_logits.shape, 0, 1).mul( tf.exp(log_std))));
    });
}

/**
 * 
 * @param {tf.Tensor} p_logits 
 * @param {tf.Tensor} act_ph 
 * @param {Number} act_dim 
 * @param {tf.Tensor} log_std 
 */
export function get_p_log_discrete(p_logits, act_ph, act_dim, log_std){
    return tf.tidy(()=>{
        let act_onehot = tf.oneHot(act_ph, depth=act_dim);
        return tf.sum(act_onehot.mul(tf.logSoftmax(p_logits), axis=-1));    
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
        //log_p = -0.5 *((x-mean)**2 / (tf.exp(log_std)**2+1e-9) + 2*log_std + np.log(2*np.pi));
        return tf.keep(tf.sum(log_p, -1));    
    });
}

function mlp(x, hidden_layers, output_size, activation='relu', last_activation='relu'){
    return tf.tidy(()=>{
        let inputt = tf.input({shape: [null, x[0]]});
        x = inputt;
        for(let l=0; l < hidden_layers.length; l++){
            x = tf.layers.dense({units:hidden_layers[l], activation:activation}).apply(x);
        }
        let output = tf.layers.dense({units: output_size[0], activation: last_activation}).apply(x);
        return tf.keep(tf.model({inputs:inputt, outputs:output}));
    });
}

export function PPO(env, agent, hidden_sizes=[32], cr_lr=5e-3, ac_lr=5e-3, num_epochs=50, minibatch_size=5000, gamma=0.99, lam=0.95, number_envs=1, eps=0.1, 
    actor_iter=5, critic_iter=10, steps_per_env=100, action_type='Discrete'){
    if (action_type === "Discrete"){
        PPODiscrete(env, hidden_sizes, cr_lr, ac_lr, num_epochs, minibatch_size, gamma, lam,number_envs, eps, actor_iter, critic_iter, steps_per_env);
    } else {
        PPOContinuous(env, hidden_sizes, cr_lr, ac_lr, num_epochs, minibatch_size, gamma, lam,number_envs, eps, actor_iter, critic_iter, steps_per_env);
    }
}

function PPODiscrete(env, agent, hidden_sizes=[32], cr_lr=5e-3, ac_lr=5e-3, num_epochs=50, minibatch_size=5000, gamma=0.99, lam=0.95, number_envs=1, eps=0.1, 
    actor_iter=500, critic_iter=500, steps_per_env=5){
    
    return;
}

export async function PPOContinuous(opt){

    const loadModels = async function(){
        const saveResult = await p_logits.save('downloads://ppo_policy');
    };


    // tf.tidy(()=>{
        let env = opt.env;
        let agent = opt.agent;
        let hidden_sizes=opt.hidden_sizes; 
        let cr_lr=opt.cr_lr; 
        let ac_lr=opt.ac_lr;
        let num_epochs=opt.num_epochs;
        let minibatch_size=opt.minibatch_size; 
        let gamma=opt.gamma;
        let lam=opt.lam;
        let number_envs=opt.number_envs; 
        let eps=opt.eps;
        let actor_iter=opt.actor_iter;
        let critic_iter=opt.critic_iter;
        let steps_per_env=opt.steps_per_env;
    
        let envs = [];
        envs.push(env);
        let agents = [];
        agents.push(agent);
        let obs_dim = agents[0].observation_space.shape;
    
        let low_action_space = agents[0].action_space.low
        let high_action_space = agents[0].action_space.high
        let act_dim = agents[0].action_space.shape
    
        // console.log(act_dim, obs_dim);
        let p_logits = mlp(obs_dim, hidden_sizes, act_dim, 'tanh', 'tanh');
        let log_std = tf.variable(tf.fill(act_dim, -0.5), false, 'log_std');
        
        let p_noisy = get_p_noisy;
        let act_smp = act_smp_cont;
        let p_log = get_p_log_cont;
        let s_values = mlp(obs_dim, hidden_sizes, [1], 'tanh', null);
        
        let p_opt = tf.train.adam(ac_lr);
        let v_opt = tf.train.adam(cr_lr);
        
        let step_count = 0;
        // console.log("Num epochs ", num_epochs);
        for(let ep=0; ep<num_epochs;ep++){
            let buffer = new Buffer_a(gamma, lam);
            let batch_rew = [];
            let batch_len = [];        
            for(let env of envs){
                let temp_states = [];
                let temp_rewards = [];
                let temp_actions = [];
                let temp_values = [];
                for(let i=0; i < steps_per_env; i++){
                    // console.log(i);
                    let nobs = tf.tensor([env.n_obs]);
                    let nobs_e = tf.expandDims(nobs, 0);
                    let p_logits_val = p_logits.apply(nobs);
                    let p_noisy_val = p_noisy(p_logits_val, log_std);
                    let act1 = act_smp(p_noisy_val, low_action_space, high_action_space);
                    let val = s_values.apply(nobs);
                    let act = tf.squeeze(act1);
                    env.action = act.arraySync();
                    let [obs2, rew, done, _] = await env.step();
                    // console.log("after await");
                    temp_states.push([env.n_obs.slice()])
                    temp_rewards.push([rew]);
                    temp_actions.push([act.arraySync()]);
                    let squeezed_val = tf.squeeze(val);
                    temp_values.push([squeezed_val.arraySync()]);
                    env.n_obs = obs2.slice()
                    step_count += 1
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
                    p_logits_val.dispose();
                    nobs_e.dispose();
                    p_noisy_val.dispose();
                    val.dispose();
                    squeezed_val.dispose();
                    act1.dispose();
                    act.dispose();
                }

                let nobs = tf.tensor([env.n_obs]);
                nobs = tf.expandDims(nobs, 0);
                let last_v = s_values.apply(nobs);
                buffer.store(temp_states, temp_rewards, temp_actions, temp_values, last_v);
                // for(let i = 0; i < temp_states.length; i++){
                //     temp_states[i].dispose();
                // }
                // for(let i = 0; i < temp_actions.length; i++){
                //     temp_actions[i].dispose();
                // }
                // for(let i = 0; i < temp_values.length; i++){
                //     temp_states[i].dispose();
                // }

            }
            // console.log("on end");
// CHECKING Stopped HERE!            

        let [obs_batch, act_batch, adv_batch, rtg_batch] = buffer.get_batch();
        let p_logits_value = p_logits.apply(obs_batch);
        let old_p_log = p_log(act_batch, p_logits_value, log_std);
        let old_p_batch = old_p_log;

        let lb = buffer.len();
        let shuffled_batch = tf.util.createShuffledIndices(lb); 
        
        for(let j=0; j < actor_iter; j++){
            tf.util.shuffle(shuffled_batch);
            for(let idx = 0; idx < lb; idx += minibatch_size){
                let minib = shuffled_batch.slice(idx, Math.min(idx+minibatch_size,lb));
                // let old_p_log_ph = old_p_batch[minib];
                let gat_tensor = tf.tensor(new Int32Array(minib));
                let p_loss = ()=> tf.tidy(()=>{
                    let p_logits_value = p_logits.apply(obs_batch.gather(gat_tensor));
                    let p_log_value = p_log( act_batch.gather(gat_tensor), p_logits_value, log_std)
                    let p_loss_v = clipped_surrogate_obj(p_log_value, old_p_batch.gather(gat_tensor), adv_batch.gather(gat_tensor), eps)
                    return p_loss_v;    
                });
                
                let gradients = tf.variableGrads(p_loss, p_logits.getWeights());
                p_opt.applyGradients(gradients.grads);
                tf.dispose(gradients);
                tf.dispose(gat_tensor);
            }

        }

        for(let j=0; j < critic_iter; j++){
            tf.util.shuffle(shuffled_batch);
            for(let idx = 0; idx < lb; idx += minibatch_size){
                let minib = shuffled_batch.slice(idx, Math.min(idx+minibatch_size,lb));
                let gat_tensor = tf.tensor(new Int32Array(minib) );
                let v_loss = ()=> tf.tidy(()=>{
                    let s_v = tf.squeeze(s_values.apply(obs_batch.gather(gat_tensor)));
                    let v_loss_v = tf.mean(rtg_batch.gather(gat_tensor).sub(s_v).square());
                    return v_loss_v;
                });

                let  gradients = tf.variableGrads(v_loss, s_values.getWeights());
                v_opt.applyGradients(gradients.grads);
                tf.dispose(gradients);
                tf.dispose(gat_tensor);
            }    
            

        }
        obs_batch.dispose();
        act_batch.dispose();
        adv_batch.dispose();
        rtg_batch.dispose();

        // loadModels();
        }
    // });

}

