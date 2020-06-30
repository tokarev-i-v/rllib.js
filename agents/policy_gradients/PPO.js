import * as tf from '@tensorflow/tfjs-node-gpu';

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
    let rt = new_p.sub(old_p).exp();
    let rtmul = rt.mul(adv);
    let clipped = tf.clipByValue(rt, 1-eps, 1+eps).mul(adv);
    let minimum = tf.minimum(rtmul, clipped);
    let meaned = minimum.mean();
    return -meaned.dataSync();
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
    store(temp_traj, last_sv){
        if (temp_traj.length > 0){
            tf.tidy(()=>{
                let tens = tf.tensor(temp_traj);
                this.ob = tf.keep(this.ob.concat(tens.slice([0], [temp_traj.length,1]).flatten()));
                let rtg = discounted_rewards(tens.slice([0,1], [temp_traj.length,1]).flatten(), last_sv, this.gamma);
                this.adv = tf.keep(this.adv.concat(GAE(tens.slice([0,1], [temp_traj.length,1]).flatten(), tens.slice([0,3], [temp_traj.length,1]).flatten(), last_sv, this.gamma, this.lam)));
                this.rtg = tf.keep(this.rtg.concat(rtg));
                this.ac = tf.keep(this.ac.concat(tens.slice([0,2], [temp_traj.length,1]).flatten()));    
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
        let log_p = x.sub(mean).square().div(log_std.exp().square().add(tf.scalar(1e-9))).add(log_std.mul(tf.scalar(2))).add(tf.scalar(2*Math.PI).log()).mul(tf.scalar(-0.5))
        //log_p = -0.5 *((x-mean)**2 / (tf.exp(log_std)**2+1e-9) + 2*log_std + np.log(2*np.pi));
        return tf.keep(tf.sum(log_p, -1));    
    });
}

export function PPO(env_name, hidden_sizes=[32], cr_lr=5e-3, ac_lr=5e-3, num_epochs=50, minibatch_size=5000, gamma=0.99, lam=0.95, number_envs=1, eps=0.1, 
    actor_iter=5, critic_iter=10, steps_per_env=100, action_type='Discrete'){
    if (action_type === "Discrete"){
        PPODiscrete(env_name, hidden_sizes, cr_lr, ac_lr, num_epochs, minibatch_size, gamma, lam,number_envs, eps, actor_iter, critic_iter, steps_per_env);
    } else {
        PPOContinuous(env_name, hidden_sizes, cr_lr, ac_lr, num_epochs, minibatch_size, gamma, lam,number_envs, eps, actor_iter, critic_iter, steps_per_env);
    }
}

function PPODiscrete(env_name, hidden_sizes=[32], cr_lr=5e-3, ac_lr=5e-3, num_epochs=50, minibatch_size=5000, gamma=0.99, lam=0.95, number_envs=1, eps=0.1, 
    actor_iter=5, critic_iter=10, steps_per_env=100){
    
    return;
}

function PPOContinuous(env_name, hidden_sizes=[32], cr_lr=5e-3, ac_lr=5e-3, num_epochs=50, minibatch_size=5000, gamma=0.99, lam=0.95, number_envs=1, eps=0.1, 
    actor_iter=5, critic_iter=10, steps_per_env=100){
        
        return;

}

