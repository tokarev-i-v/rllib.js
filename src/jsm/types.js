import * as tf from '@tensorflow/tfjs-node'

/**
 * 
 */
export class Buffer{
    constructor(minsize, size){
      this.v = [];
      this.size = typeof(size)==='undefined' ? 100 : size;
      this.minsize = typeof(minsize)==='undefined' ? 20 : minsize;
      this.sum = 0;
    }
    add(x) {
      this.v.push(x);
      this.sum += x;
      if(this.v.length>this.size) {
        var xold = this.v.shift();
        this.sum -= xold;
      }
    }
    get_average() {
      if(this.v.length < this.minsize) return -1;
      else return this.sum/this.v.length;
    }
    reset(x) {
      this.v = [];
      this.sum = 0;
    }
  }
  
  
  /**
   * Class describe continuous R^n space.
   */
export class BoxSpace {
    constructor(low=-1, high=1, shape=3){
      this.low = low;
      this.high = high;
      this.shape = shape;
      this.value = tf.tensor(Array(shape[0]).fill(0));
    }
    /**
     * @returns {tf.tensor1d}
     */
    clip(val){
      return tf.tidy(()=>{
        return tf.keep(tf.clipByValue(val, this.low, this.high));
      });
    }
  
    /**
     * @returns {tf.tensor1d}
     */  
    sample(){
      return tf.tidy(()=>{
        return tf.keep(tf.randomUniform(this.shape));
      });
    }
  }