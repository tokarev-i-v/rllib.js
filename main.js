import * as tf from '@tensorflow/tfjs'
import DQN from "./agents/DQN/BaseDQN";
import FlatAreaEatWorld from "./envs/FlatAreaWorld/FlatAreaEatWorld_d"
let world = new FlatAreaEatWorld({agent: DQN});
tf.disableDeprecationWarnings();
tf.setBackend("cpu").then(()=>{
    world.start();
});