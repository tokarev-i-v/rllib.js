/**
 * We need to create gl context.
 * And set gl.canvas object with 'width', 'height' properties.
 * headless-gl used by three.js.
 */
GLOBAL.gl = require('gl')(1,1); //headless-gl
GLOBAL.gl.canvas = {
  "width": 1,
  "height": 1
}
import moment from 'moment';
import fs from 'fs'
import {Worker, workerData, MessageChannel} from 'worker_threads';
import {JSDOM} from 'jsdom';
/** Create DOM structure. */
const jsdel =  new JSDOM(`<!DOCTYPE html><html><head></head><body>hello</body></html>`);
global.window = jsdel.window;
global.document = jsdel.window.document;

import {HungryWorld2D, Agent as HungryAgent} from "./src/jsm/envs/HungryWorld2D/HungryWorld2D_node.js";
import {build_full_connected}  from './src/jsm/neuralnetworks_node.js';
import {getWeightsFromModelToWorkerTransfer, setWeightsToModelByObject}  from './src/jsm/utils_node.js';
let curretWorldClass = HungryWorld2D;

function runService(workerData) {
    return new Promise((resolve, reject) => {

      var PPOworker = new Worker("./src/jsm/agents/policy_gradients/ppo_node_worker.js", {workerData});

      var a = new HungryAgent2D({eyes_count: 10});
      let cur_nn = build_full_connected(a.observation_space.shape, [128, 128], a.action_space.shape, 'tanh', 'tanh');
      let weights_obj = getWeightsFromModelToWorkerTransfer(cur_nn);
      var w = new curretWorldClass({});
      w.addAgent(a);
      PPOworker.on('message', function(data){
          if(data.msg_type === "step"){
              var step_data = w.step(data.action);
              PPOworker.postMessage({
                msg_type: "step", 
                step_data: step_data, 
                n_obs: w.n_obs, 
                e_r: w.get_episode_reward(), 
                e_l: w.get_episode_length()
              });
          }
          if(data.msg_type === "get_policy_weights"){
              let curr_time_str = moment().format();
              let dir = "./PPO/" + curr_time_str + "/";
              if (!fs.existsSync(dir)){
                fs.mkdirSync(dir);
              }
              let model = build_full_connected(a.observation_space.shape, [64,64], a.action_space.shape, 'tanh', 'tanh');
              model = setWeightsToModelByObject(model, data.policy_weights);
              model.save('file://' + dir + 'model');
          }
      });

      PPOworker.on('error', reject);
      PPOworker.on('exit', (code) => {
        if (code !== 0){
          reject(new Error(`Worker stopped with exit code ${code}`));
        }
      });
      /*Starting training process*/
      PPOworker.postMessage({
        msg_type: "start",
        observation_space: a.observation_space,
        action_space: a.action_space,
        n_obs: w.n_obs,
        policy_nn: weights_obj,
        num_epochs: 60
      });
      
    })
  }
  
  async function run() {
    const result = await runService('world');
    console.log(result);
  }

  run().catch(err => console.error(err))