# rllib.js
Reinforcement learning library with JavaScript.  
This library provides some reinforcement learning algorithms and environments.  
At this moment uses [TensorFlow.js](!https://github.com/tensorflow/tfjs) to implement functions and [three.js](!https://threejs.org/) for visualization.  
Live example: https://polyzer.github.io/rllib/build/ppo_web_worker.html

## Install and start example
Starting example:

### Linux  
```bash
git clone https://github.com/polyzer/rllib.js.git  
npm i   
npm run ppo
```

### Windows  
```bash
git clone https://github.com/polyzer/rllib.js.git  
npm i   
npm run ppo
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
