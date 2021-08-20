# rllib.js
Reinforcement learning library with JavaScript.  
It provides some reinforcement learning algorithms and environments.  
At this moment uses [TensorFlow.js](!https://github.com/tensorflow/tfjs) to implement functions and [three.js](!https://threejs.org/) for visualization.  
[Live example](https://polyzer.github.io/rllib/build/ppo_web_worker.html)
 

## Motivation
**JavaScript** is a very popular programming language but JS developers have no instruments of RL.  
This library was created to correct the situation.
Web game makers can use it to train game bots in 2D and 3D environments.  
Other can find it useful to make experiments.  

Unity developers can use ML methods with [ml-agents](https://github.com/Unity-Technologies/ml-agents)

## Install and start example
Starting example:

### Linux  
```bash
git clone https://github.com/polyzer/rllib.js.git  
npm i   
npm run ppo_2d
```

### Windows  
```bash
git clone https://github.com/polyzer/rllib.js.git  
npm i   
npm run ppo_2d
```
There, please follow in:
```bash
./dist/ppo_worker_main*.js
```  
and replace
```javascript
var PPOworker = new Worker("/agents\policy_gradients\ppo_worker.js");
```  
to
```javascript
var PPOworker = new Worker("/agents/policy_gradients/ppo_worker.js");
```  
[this issue](https://github.com/parcel-bundler/parcel/issues/1990)

### Node.js
You can start Node.js example with:
```bash
npm run ppo_node
```


## Algorithms:  
Deep Q-learning:  
- [x] Base Deep Q-learning  
- [ ] Dueling Q-learning  
- [ ] Double Q-learning  

Policy gradient:  
- [ ] REINFORCE  
- [ ] Actor-Critic  
- [x] PPO  
- [ ] TRPO  

## Environments:
Without physics:  
Flat world 3D. 
Discrete and continuous versions.
* Agent have 10 eye-detectors.  
* Target: Agent trying to learn eating only green items if he see them.  
* Actions: turns on specified angles.  
* Space: Agent recieves signal from each eye that specifies type of saw item.  



Visual library: [three.js](!https://threejs.org/)

![](./readme/output.gif)
