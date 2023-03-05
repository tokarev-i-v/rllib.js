# rllib.js
Reinforcement learning library with JavaScript.  
It provides some reinforcement learning algorithms and environments.  
At this moment uses [TensorFlow.js](!https://github.com/tensorflow/tfjs) to implement functions and [three.js](!https://threejs.org/) for visualization.

## [Live example](https://tokarev-i-v.github.io/rllib/build/ppo-threejs-hungry-jsm.html)
 
Visual library: [three.js](!https://threejs.org/)

![](./readme/output.gif)

## Motivation
**JavaScript** is a very popular programming language but JS developers have no instruments of RL.  
This library was created to correct the situation.
Web game makers can use it to train game bots in 2D and 3D environments.  
Other can find it useful to make experiments.  

Unity developers can use ML methods with [ml-agents](https://github.com/Unity-Technologies/ml-agents)

## Install and start example
Starting example:

### Windows
On Windows you need preinstalled MS Visual Studio Community edition for build 'gl' library.

### Linux, Windows
```bash
git clone https://github.com/polyzer/rllib.js.git  
npm i   
```

### Hungry world 2D example using PPO algorithm in js
```bash
npm run ppo-hungry-js
```

### Hungry world 2D example using PPO algorithm with Parcel
```bash
npm run ppo-hungry-jsm
```

### Hungry world example using PPO algorithm with Parcel
```bash
npm run ppo-hungry3d-jsm
```


### Hungry world example using PPO algorithm Node.js training
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

## Known issues:
### Node.js
[symbol lookup error on nVidia](https://github.com/stackgl/headless-gl/issues/65)

It could be helpful:
```bash
npm rebuild --build-from-source gl
```
