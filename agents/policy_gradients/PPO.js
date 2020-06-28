import * as tf from '@tensorflow/tfjs-node-gpu';

/**
 * 
 * @param {tf.Tensor} rews 
 * @param {number} last_sv 
 * @param {number} gamma 
 */
const discounted_rewards = (rews, last_sv, gamma) => tf.tidy(()=>{
    let rtg = rews.bufferSync();
    
    rtg.set(rews[rews.length-1] + gamma * last_sv, rews.length-1);
    for(let i = rews.length-1;i >= 0; i--){
        rtg.set(rews[i] + gamma * rtg[i+1], i);
    }
    return rtg.toTensor();
});
/**
 * 
 * @param {tensor} rews 
 * @param {*} v 
 * @param {*} v_last 
 * @param {*} gamma 
 * @param {*} lam 
 */
const GAE = (rews, v, v_last, gamma=0.99, lam=0.95) => tf.tidy(()=>{
    let vs = np.append(v, v_last)
    let delta = np.array(rews) + gamma*vs[1:] - vs[:-1]
    let gae_advantage = discounted_rewards(delta, 0, gamma*lam)
    return gae_advantage
});
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
                this.ob = this.ob.concat(tens.slice([0], [temp_traj.length,1]).flatten());
                let rtg = discounted_rewards(tens.slice([0,1], [temp_traj.length,1]).flatten(), last_sv, this.gamma);
                this.adv = this.adv.concat(GAE(tens.slice([0,1], [temp_traj.length,1]).flatten(), tens.slice([0,3], [2,1]).flatten(), last_sv, this.gamma, this.lam));
                this.rtg = this.rtg.concat(rtg);
                this.ac = this.ac.concat(tens.slice([0,2], [temp_traj.length,1]).flatten());    
            });
        }
    }
    get_batch(){
        let adv_t = tf.tensor(this.adv); 
        let norm_adv = (adv_t - adv_t.mean()) / (tf.moments(adv_t).variance.sqrt() + 1e-10)
        return tf.tensor(this.ob), tf.tensor(this.ac), tf.tensor(norm_adv), tf.tensor(this.rtg)
    }
    len(){
        return this.ob.shape[0];
    }
}