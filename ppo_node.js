GLOBAL.gl = require('gl')(1,1); //headless-gl
GLOBAL.gl.canvas = {
  "width": 1,
  "height": 1
}
import moment from 'moment';
import fs from 'fs'
import {Worker, workerData, MessageChannel} from 'worker_threads';
import {JSDOM} from 'jsdom';
const jsdel =  new JSDOM(`<!DOCTYPE html><html><head></head><body>hello</body></html>`);
global.window = jsdel.window;
global.document = jsdel.window.document;
// tf.disableDeprecationWarnings();

import {HuntersWorld, Agent as HunterAgent} from "./src/jsm/envs/HuntersWorld/HuntersWorld_node.js";
import {build_full_connected}  from './src/jsm/neuralnetworks_node.js';
import {getWeightsFromModelToWorkerTransfer, setWeightsToModelByObject}  from './src/jsm/utils_node.js';
let curretWorldClass = HuntersWorld;

function runService(workerData) {
    return new Promise((resolve, reject) => {

      var PPOworker = new Worker("./src/jsm/agents/policy_gradients/ppo_node_worker.js", {workerData});

      var a = new HunterAgent({eyes_count: 10});
      let cur_nn = build_full_connected(a.observation_space.shape, [128, 128], a.action_space.shape, 'tanh', 'tanh');
      let weights_obj = getWeightsFromModelToWorkerTransfer(cur_nn);
      // let ui = new SimpleUI({parent: document.body, policy_nn: cur_nn, worker: PPOworker});
      var w = new curretWorldClass({});
      //cur_nn = tf.loadLayersModel('http://localhost:1234/models/mymodel.json');
      w.addAgent(a);
      PPOworker.on('message', function(data){
          if(data.msg_type === "step"){
              var step_data = w.step(data.action);
              // console.log("step data:", step_data);
              PPOworker.postMessage({msg_type: "step", step_data: step_data, n_obs: w.n_obs, e_r: w.get_episode_reward(), e_l: w.get_episode_length()});
              // resolve();
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