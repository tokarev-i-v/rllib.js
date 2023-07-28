/**
 * Implementation in progress.
 */

export class Buffer{
    constructor(gamma = 0.99){
        this.gamma = gamma;
        this.obs = [];
        this.act = [];
        this.ret = [];
        this.rtg = [];
    }

    store(temp_traj, last_sv){
        if(temp_traj.length > 0){
            let nulls = temp_traj.map((el)=>{
                return el[0];
            });
            let firsts = temp_traj.map((el)=>{
                return el[1];
            });
            let seconds = temp_traj.map((el)=>{
                return el[2];
            });
            let thirds = temp_traj.map((el)=>{
                return el[3];
            });
            this.obs = this.obs.concat(nulls);
            let rtg = this.discounted_rewards(firsts, last_sv, self.gamma);
            let temp = rtg.map((value, index) => {
                value - thirds[i];
            });
            this.ret = this.ret.concat(temp);
            this.rtg.concat(rtg);
            this.act.concat(seconds);
        }
    
    }

    getBatch(){
        return tf.tensor(this.obs), tf.tensor(this.act), tf.tensor(this.ret), tf.tensor(this.rtg);
    }

    discounted_rewards(rews, last_sv, gamma){
        let rtg = tf.zerosLike(rews).buffer();
        rtg.set(rews[rews.length-1] + gamma * last_sv, rews.length-1);
        for(let i = rews.length-1;i >= 0; i--){
            rtg.set(rews[i] + gamma * rtg[i+1], i);
        }

        return rtg.toTensor().dataSync();
    }

    get length(){
        return this.obs.length;
    }
}

export class AC{

    constructor(hidden_sizes=[64], ac_lr=0.004, cr_lr=0.015, gamma=0.99, steps_per_epoch=100, ){
    }
    log_summary(writer, step, p_loss, entropy, p_log, ret_batch){
    }
    mlp(x, hidden_layers, output_size, activation= tf.layers.reLU, last_activation=null){
        //multilayer perceptron
        let inputt = tf.input({shape: [null, x[0]]});
        x = inputt;
        for (let l in hidden_layers){
            x = tf.layers.dense({units: l, activation: activation}).apply(x);
        }
        let output = tf.layers.dense({units: output_size, activation: last_activation}).apply(x);
        return tf.model({inputs: inputt, outputs: output})
    }
    
    softmax_entropy(values){
        return tf.sum(values.softmax() * values.logSoftmax(), axis=-1);
    }
    


}
