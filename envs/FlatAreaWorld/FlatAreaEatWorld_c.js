// import * as tf from '@tensorflow/tfjs-node-gpu';
import * as tf from '@tensorflow/tfjs';
import * as THREE from '../../src/jsm/three.module';
import {ColladaLoader} from '../../src/jsm/ColladaLoader';
import {FlyControls} from '../../src/jsm/FlyControls';
import Stats from '../../src/jsm/stats.module';
var canvas, ctx;

function getRandomArbitrary(min, max) {
  return Math.random() * (max - min) + min;
}

function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min)) + min; //Максимум не включается, минимум включается
}


/**
 * Class describe continuous R^n space.
 */
class BoxSpace {
  constructor(low=-1, high=1, shape=3){
    this.low = low;
    this.high = high;
    this.shape = shape;
    this.value = tf.tensor(Array(shape[0]).fill(0));
  }
  control(){
    return tf.tidy(()=>{
      return tf.keep(tf.clipByValue(rt, this.low, this.high));
    });
  }
  sample(){
    return tf.tidy(()=>{
      return tf.keep(tf.randomUniform(this.shape));
    });
  }
}

class Food {
  constructor(pos){
    this.rad = 1;
    this._view = new THREE.Mesh(
      new THREE.SphereBufferGeometry(this.rad,this.rad,this.rad),
      new THREE.MeshBasicMaterial({color: 0x11FF11})
    )
    this.age = 0;
    this.type = 1;
    this.cleanup_ = false;
    this._view._rl = {
      type: this.type
    }
    this._view.position.copy(pos);
  }
  get view(){
    return this._view;
  }
  get position(){
    return this._view.position;
  }
  set position(vec){
    this._view.position.copy(vec);
  }
}

class Poison {
  constructor(pos){
    this.rad = 1;
    this._view = new THREE.Mesh(
      new THREE.SphereBufferGeometry(this.rad,this.rad,this.rad),
      new THREE.MeshBasicMaterial({color: 0xFFF422})
    )      
    this.age = 0;
    this.type = 2;
    this.cleanup_ = false;
    this._view.position.copy(pos);
    this._view._rl = {
      type: this.type
    }
  }
  get view(){
    return this._view;
  }
  get position(){
    return this._view.position;
  }
  set position(vec){
    this._view.position.copy(vec);
  }
}

/**
 * Agent from this world;
 */
export class Agent{
  /**
   * 
   * @param {Object} opt 
   */
  constructor(opt){
    this.rad = 2;
    this._view = new THREE.Mesh(
      new THREE.BoxBufferGeometry(this.rad,this.rad,this.rad),
      new THREE.MeshBasicMaterial({color: 0xFFAA11})
    );
    

    this.min_action = -1.0;
    this.max_action = 1.0;

    this.action_space = new BoxSpace(this.min_action,this.max_action, [3]);
    this.eyes_count = opt.eyes_count;
    this.observation_space = new BoxSpace(-10000000, 100000000, [this.eyes_count * 3])
    console.log("Observation space shape: ", this.observation_space.shape);
    this.eyes = [];
    let r = 20;
    let alpha = -30;
    /**Now we create agent's eyes*/
    for (let i = 0; i < this.eyes_count; i++){
        let eye = new Eye(this, alpha, r);
        let mesh = eye.view;
        this.view.add(mesh);
        this.eyes.push(eye);
        alpha += 20;
    }
    this._frontEye = null;
    if(this.eyes.length % 2 === 0){
      this._frontEye = this.eyes[Math.round(this.eyes.length/2)];
    }else {
      this._frontEye = this.eyes[Math.round(this.eyes.length/2)-1];
    }
    if (opt.algo){
      this.brain = new opt.algo({num_states: this.eyes.length * 3, num_actions: this.action_space.length});
    }
    
    this.reward_bonus = 0.0;
    this.digestion_signal = 0.0;
    // outputs on world
    this.rot1 = 0.0; // rotation speed of 1st wheel
    this.rot2 = 0.0; // rotation speed of 2nd wheel
    this.speed = 0.0;
  }
  
  sample_actions(){
    return this.action_space.sample();
  }

  forward() {
    // in forward pass the agent simply behaves in the environment
    // create input to brain
    var num_eyes = this.eyes.length;
    var input_array = new Array(num_eyes * 3);
    for(var i=0;i<num_eyes;i++) {
      var e = this.eyes[i];
      input_array[i*3] = 1.0;
      input_array[i*3+1] = 1.0;
      input_array[i*3+2] = 1.0;
      if(e.sensed_type !== -1) {
        // sensed_type is 0 for wall, 1 for food and 2 for poison.
        // lets do a 1-of-k encoding into the input array
        input_array[i*3 + e.sensed_type] = e.sensed_proximity/e.max_range; // normalize to [0,1]
      }
    }
    return input_array;
    
  }

  backward() {
    // in backward pass agent learns.
    // compute reward 
    var proximity_reward = 0.0;
    var num_eyes = this.eyes.length;
    for(var i=0;i<num_eyes;i++) {
      var e = this.eyes[i];
      // agents dont like to see walls, especially up close
      // proximity_reward += e.sensed_type === 0 ? e.sensed_proximity/e.max_range : 0.0;
      proximity_reward += e.sensed_type === 1 ? 1 - e.sensed_proximity : 0.0;
      // proximity_reward += e.sensed_type === 2 ? -(1 - e.sensed_proximity) : 0.0;
    }
    // console.log("num_eyes: %s ", num_eyes);    
    proximity_reward = proximity_reward/num_eyes;
    
    // agents like to go straight forward
    var forward_reward = 0.0;
    if(this.actionix === 0 && proximity_reward > 0.75) forward_reward = 0.1 * proximity_reward;
    
    // agents like to eat good things
    var digestion_reward = this.digestion_signal;
    this.digestion_signal = 0.0;
    var reward = proximity_reward + forward_reward + digestion_reward; 
    // console.log("dig: %s proximity_reward %s forward_reward %s reward: %s", digestion_reward, proximity_reward, forward_reward, reward);     
    // pass to brain for learning
    //this.brain.backward(reward);
    return reward;
  }

  get view(){
    return this._view;
  }
  get position(){
    return this._view.position;
  }
  set position(vec){
    this._view.position.copy(vec);
  }

  get rotation(){
    return this._view.rotation;
  }
  set rotation(vec){
    this._view.rotation.copy(vec);
  }
  
  get angle(){
    return this._view.rotation.z;
  }
  set angle(val){
    this._view.rotation.z = val;
  }
  get frontEye(){
    return this._frontEye;
  }
}
/**
 * @class Eye
 * It presents as agent's eye detector.
 */
export class Eye{
  /**
   * 
   * @param {THREE.Vector3} agent_pos_vec Vector that would use as src
   * vector for raycastring
   * @param {Number} alpha angle 
   * @param {Number} r radius
   */
  constructor(a, alpha, r){
    this._view = new THREE.Mesh(
      new THREE.BoxBufferGeometry(1,1,10),
      new THREE.MeshBasicMaterial({color: 0x111111})
    );
    /**setting Eye position and rotation */
    this._view.position.x = Math.sin(Math.PI*alpha/180)*r;
    this._view.position.y = Math.cos(Math.PI*alpha/180)*r;
    this._view.rotation.z = Math.PI*(180-alpha)/180;
    this._view.geometry.computeBoundingBox();
    this.raycaster = new THREE.Raycaster();
    this.max_range = 20;
    this.sensed_proximity = 20; // what the eye is seeing. will be set in world.tick()
    this.a = a;
  }
  get view(){
    return this._view;
  }
  /**
   * This function return the nearest detected object.
   * @param {THREE.Mesh[]} targets array of intersection targets
   * @returns {Object|null} 
   */
  getNearestCollision(targets_objs){
    let targets = targets_objs.map((el)=>{
        return el.view;
    });
    let dst = new THREE.Vector3();
    dst.setFromMatrixPosition( this._view.matrixWorld );
    dst.add(this.a.position.clone().negate());
    dst.normalize();
    this.raycaster.set(this.a.position, dst);
    let intersects = this.raycaster.intersectObjects(targets);
    if (intersects.length > 0 && intersects[0].distance < this.max_range){
      return {obj: intersects[0].object, type: intersects[0].object._rl.type, dist: intersects[0].distance}
    } else {
      return null;
    }
  }
}
  
  class Wall {
    constructor(p1, p2){
      this._view = new THREE.Mesh(
        new THREE.CubeBufferGeometry(),
        new THREE.MeshBasicMaterial({color: 0xf2f2f2})
      );
      this.type = 0;
    }
    get boundingBox(){
      this._view.geometry.computeBoundingBox();
      return this._view.geometry.boundingBox;
    }
  }
  
  
  var Item = THREE.Object3D;
  /**
   * @class
   * World Contains all features.
   */
export class FlatAreaEatWorld_c {
    constructor(opt){


      this.init();
      this.agents = [];

      this.W = 200;
      this.H = 200;

      this.clock = 0;
      
      this.walls = []; 
      var pad = 10;

      this.rew_episode = 0;
      this.len_episode = 0;

      // set up food and poison
      this.items = []
      for(var k=0;k<4000;k++) {
        this.generateItem();
      }
      if(opt){
        if(opt.algo){
          let agent = new Agent({eyes_count: 10, algo: opt.agent});
          this.Scene.add(agent.view);
          this.agents.push(agent);  
        }
      
      }
      this.render();
    }   

    addAgent(agent){
      this.Scene.add(agent.view);
      this.agents.push(agent);
      this.n_obs = this.agents[0].forward();
    }

    generateItem(){
      var x = getRandomArbitrary(-this.W, this.W);
      var y = getRandomArbitrary(-this.H, this.H);
      var t = getRandomInt(1, 3); // food or poison (1 and 2)
      if (t == 1){
        var it = new Food(new THREE.Vector3(x, y, 0));
      }
      else{
        var it = new Poison(new THREE.Vector3(x, y, 0));
      }
      this.items.push(it);
      this.Scene.add(it.view);
    }

    init(json_params){
      // if(typeof(document) !== typeof(undefined)){
        this.Container = document.createElement("div");
        this.Container.id = "MainContainer";
        this.Container.classList.add("Container");
        
        this.Renderer = new THREE.WebGLRenderer();
        this.Renderer.setSize(window.innerWidth, window.innerHeight);
        this.Container.appendChild(this.Renderer.domElement);
  
        document.body.insertBefore( this.Container, document.body.firstChild);
  
        this.stats = new Stats();
        document.body.appendChild(this.stats.dom);

        this.Camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 10000);

        this.Controls = new FlyControls(this.Camera, document.getElementById("MainContainer"));
        this.Controls.movementSpeed = 13;
        this.Controls.rollSpeed = Math.PI / 8;
        this.Controls.autoForward = false;
        this.Controls.dragToLook = false;
  
      // } else {
      //   this.Camera = new THREE.PerspectiveCamera(75, 1, 0.1, 10000);
      // }


      this.Camera.position.set(0,0, 10);
      this.Scene = new THREE.Scene();
      this.Scene.background = new THREE.Color( 0xaaccff );
      this.Scene.fog = new THREE.FogExp2( 0xaaccff, 0.007 );

      this.Loader = new ColladaLoader();

      this.AmbientLight = new THREE.AmbientLight(0xFFFFFF, 0.9);
      this.Scene.add(this.AmbientLight);

      
      this.Clock = new THREE.Clock();

      if(typeof(document) !== typeof(undefined)){
        let TextureLoader = new THREE.TextureLoader();
        TextureLoader.load("grass.png", function (tex) {
            tex.wrapS = THREE.RepeatWrapping;
            tex.wrapT = THREE.RepeatWrapping;
            tex.repeat.set(100, 100);
            let ground = new THREE.Mesh(new THREE.PlaneBufferGeometry(1000, 1000), new THREE.MeshBasicMaterial({map: tex, side:THREE.DoubleSide}));
            //ground.rotation.x -= Math.PI/2;
            this.Scene.add(ground);
        }.bind(this));
  
      }
    }

    reset(){
        this.n_obs = this.agents[0].forward();
        this.rew_episode = 0;
        this.len_episode = 0;
        this.clock = 0;
        return this.n_obs.slice()
    }

    get_episode_reward(){
      return this.rew_episode
    }

    get_episode_length(){
      return this.len_episode
    }


    render () {
      requestAnimationFrame(this.render.bind(this));
      this.stats.update();
      
      this.Renderer.render(this.Scene, this.Camera);
      var delta = this.Clock.getDelta();
      
      this.Controls.update(delta);
    }

    // helper function to get closest colliding walls/items
    stuff_collide_(eye, check_walls, check_items) {
      var minres = false;

      if(check_walls) {
          let res = eye.getNearestCollision(this.walls);
          if(res) {
            res.type = 0; // 0 is wall
            if(!minres) { minres=res; }
          }
      }
      // collide with items
      if(check_items) {
        let res = eye.getNearestCollision(this.items);
        if(res) {
          if(!minres) { minres=res; }
        }
      }
      return minres;
    }
    removeItem(it){
      this.Scene.remove(it.view);
      this.items.splice(this.items.indexOf(it), 1);
    }

    async step(action) {
      // if(!this.skipdraw || this.clock % 50 === 0) {
      //   this.render();
      // }  
      this.agents[0].rot1 = action[0];
      this.agents[0].rot2 = action[1];
      this.agents[0].speed = action[2];  
      let ret = this.tick(action);
      this.rew_episode += ret[1];
      this.len_episode += 1;
      return ret;
    }


    start() {
      requestAnimationFrame(this.start.bind(this));
      this.step();
    }

    tick(action) {

      let state, done, reward;

      // tick the environment
      this.clock++;
      
      // fix input to all agents based on environment
      // process eyes
      this.collpoints = [];
      for(var i=0,n=this.agents.length;i<n;i++) {
        var a = this.agents[i];
        for(var ei=0,ne=a.eyes.length;ei<ne;ei++) {
          var e = a.eyes[ei];
          // we have a line from p to p->eyep
          var res = this.stuff_collide_(e, true, true);
          if(res) {
            // eye collided with wall
            e.sensed_proximity = res.dist;
            e.sensed_type = res.type;
          } else {
            e.sensed_proximity = e.max_range;
            e.sensed_type = -1;
          }
        }
      }
      let states = [];
      // let the agents behave in the world based on their input
      for(var i=0,n=this.agents.length;i<n;i++) {
        states.push(this.agents[i].forward());
      }
      
      // apply outputs of agents on evironment
      for(var i=0,n=this.agents.length;i<n;i++) {
        var a = this.agents[i];
        // var v = a.position.clone();
        var v = a._view.getWorldDirection();
        v.normalize();
        v.multiplyScalar(a.speed);
        a.position.add(v);
        a.rotation.x += a.rot1;
        a.rotation.y += a.rot2;
        
        // agent is trying to move from p to op. Check walls
        var res = this.stuff_collide_(a.frontEye, true, false);
        if(res) {
          // wall collision! reset position
          a.position = a.op;
        }
        
        // handle boundary conditions
        if(a.position.x<0)a.position.x=0;
        if(a.position.x>this.W)a.position.x=this.W;
        if(a.position.y<0)a.position.y=0;
        if(a.position.y>this.H)a.position.y=this.H;
      }
      
      // tick all items
      var update_items = false;
      for(var i=0,n=this.items.length;i<n;i++) {
        var it = this.items[i];
        it.age += 1;
        
        // see if some agent gets lunch
        for(var j=0,m=this.agents.length;j<m;j++) {
          var a = this.agents[j];
          var d = a.position.distanceTo(it.position);
          if(d < it.rad + a.rad) {
            
            // wait lets just make sure that this isn't through a wall
            var rescheck = this.stuff_collide_(a.frontEye, true, false);
            if(!rescheck) { 
              // ding! nom nom nom
              if(it.type === 1) a.digestion_signal += 0.5; // mmm delicious apple
              if(it.type === 2) a.digestion_signal += -0.6; // ewww poison
              this.removeItem(it);
              i--;
              n--;
              break; // break out of loop, item was consumed
            }
          }
        }
        
        if(it.age > 5000 && this.clock % 100 === 0 && getRandomArbitrary(0,1)<0.1) {
          this.removeItem(it);
          i--;
          n--;
        }
      }
      if(this.items.length < 5000) {
        this.generateItem();
      }
      let rewards = [];
      // agents are given the opportunity to learn based on feedback of their action on environment
      for(var i=0,n=this.agents.length;i<n;i++) {
        rewards.push(this.agents[i].backward());
      }
      done = 0;
      if(this.clock > 2){
        done = true;
      }
      state = states[0];
      reward = rewards[0];
      let info = {};
      return [state, reward, done, info];
    }
  }
  
