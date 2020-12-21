/**
 * In this env agent's target learn to eat greens as soon as possible.
 * 
 * Environment characteristics:
 *  --type: continuous;
 */

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis'
import * as THREE from '../../src/jsm/threejs/three.module';
import {ColladaLoader} from '../../src/jsm/threejs/ColladaLoader';
import {FlyControls} from '../../src/jsm/threejs/FlyControls';
import Stats from '../../src/jsm/threejs/stats.module';

import {params_setter, getRandomArbitrary, getRandomInt} from '../../src/jsm/utils';
import {Buffer, BoxSpace} from '../../src/jsm/types';
import {MobileControls} from '../../src/jsm/controls';

var CONSTANTS = {
  "TYPES": {
    "BULLET": 0,
    "POISON": 1,
    "APPLE": 2
  }
};


/**
 * Class describes targets, that must be eaten by agent.
 * Charasteristics:
 *  --reward: 0.99
 */
class Food {
  /**
   * 
   * @param {THREE.Vector3} pos position of the food 
   */
  constructor(pos){
    this.rad = 1;
    this._view = new THREE.Mesh(
      new THREE.SphereBufferGeometry(this.rad,32,10),
      new THREE.MeshBasicMaterial({color: 0x11FF11})
    )
    this._view.position.x = 10;
    this.age = 0;
    this.type = 1;
    this.reward = 0.99;
    this.cleanup_ = false;
    this._view._rl = {
      type: this.type,
      obj: this
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

/**
 * Class describes targets, that must be eaten by agent.
 * Charasteristics:
 *  --reward: 0.99
 */
class Poison {
  /**
   * 
 * @param {THREE.Vector3} pos
   */
  constructor(pos){
    this.rad = 1;
    this._view = new THREE.Mesh(
      new THREE.SphereBufferGeometry(this.rad,32,10),
      new THREE.MeshBasicMaterial({color: 0xFFF422})
    )      
    this.age = 0;
    this.type = 2;
    this.reward = -0.99;
    this.cleanup_ = false;
    this._view.position.copy(pos);
    this._view._rl = {
      type: this.type,
      obj: this
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
 * Class describes bullet, that could be shotted by Agent.
 */
class Bullet {
  /**
   * 
   * @param {THREE.Vector3} pos
   * @param {THREE.Vector3} dir
   */
  constructor(pos, dir){
    this.speed = 15;
    this.dir = dir;
    this.rad = 0.2;
    this.way = new THREE.Vector3();
    this._view = new THREE.Mesh(
      new THREE.SphereBufferGeometry(this.rad,32,10),
      new THREE.MeshBasicMaterial({color: 0xFF0000})
    )      
    this._view.geometry.computeBoundingBox();
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

  update(time){
    let v = this.dir.clone();
    
    if (time){
      v.multiplyScalar(this.speed*time);
    }else {
      v.multiplyScalar(this.speed);
    }
    this.position.add(v);
    this.way.add(v);
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

    let materials = [
	    new THREE.MeshStandardMaterial( { color: 0x000000 } ), // right
	    new THREE.MeshStandardMaterial( { color: 0x000000 } ), // left
	    new THREE.MeshStandardMaterial( { map: THREE.ImageUtils.loadTexture('./hunter_black_278.png')} ), // top
	    new THREE.MeshStandardMaterial( { color: 0x000000 } ), // bottom
	    new THREE.MeshStandardMaterial( { color: 0x000000 } ), // back
	    new THREE.MeshStandardMaterial( { color: 0x000000 } )  // front
	];

    this._view = new THREE.Mesh(
      new THREE.BoxBufferGeometry(this.rad,this.rad,this.rad),
      new THREE.MultiMaterial( materials )
    );
    this.min_action = -1.0;
    this.max_action = 1.0;

    this.position.y = 1;
    this.action_space = new BoxSpace(this.min_action,this.max_action, [3]);
    this.eyes_count = opt.eyes_count;
    this.observation_space = new BoxSpace(-10000000, 100000000, [this.eyes_count * 3])
    console.log("Observation space shape: ", this.observation_space.shape);
    this.eyes = [];
    let r = 20;
    let dalpha = 10;
    let alpha = -(dalpha*this.eyes_count)/2;
    /**Now we create agent's eyes*/
    for (let i = 0; i < this.eyes_count; i++){
        let eye = new Eye(this, alpha, r);
        let mesh = eye.view;
        this.view.add(mesh);
        this.eyes.push(eye);
        alpha += dalpha;
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
    this.rot = 0.0; // rotation speed of 1st wheel
    this.speed = 0.0;
    this.average_reward_window = new Buffer(10, 1000);
    this.displayHistoryData = [];
    this.surface = { name: 'Mean reward', tab: 'Charts' };
    setInterval(this.graphic_vis.bind(this), 1000);
  }
  
  fire(){
    let dir = new THREE.Vector3(); 
    this._view.getWorldDirection(dir);
    let bullet = new Bullet(this.position.clone(), dir);
    return bullet;
  }

  /**
   * @returns {BoxSpace} sampled action
   */
  sample_actions(){
    return this.action_space.sample();
  }

  /**
   * Updates graphics
   */
  graphic_vis(){
    if (this.displayHistoryData.length > 1100){
      this.displayHistoryData.splice(0,100);
    }
    this.displayHistoryData.push({"x": this.age, "y": this.average_reward_window.get_average()});
    let data = {values: this.displayHistoryData};
    tfvis.render.linechart(this.surface, data);
  }
  /**
   * 
   */
  get_observation() {
    let num_eyes = this.eyes.length;
    let obs = new Array(num_eyes * 3);
    for(let i=0;i<num_eyes;i++) {
      let e = this.eyes[i];
      obs[i*3] = 1.0;
      obs[i*3+1] = 1.0;
      obs[i*3+2] = 1.0;
      if(e.sensed_type !== -1) {
        // sensed_type is 0 for wall, 1 for food and 2 for poison.
        // lets do a 1-of-k encoding into the input array
        obs[i*3 + e.sensed_type] = e.sensed_proximity/e.max_range; // normalize to [0,1]
      }
    }
    return obs;
  }

  get_reward() {
    // compute reward 
    var proximity_reward = 0.0;
    var timed_reward = -1.0;
    var num_eyes = this.eyes.length;
    for(var i=0;i<num_eyes;i++) {
      var e = this.eyes[i];
      // Here could be
      // proximity_reward += e.sensed_type === 0 ? e.sensed_proximity/e.max_range : 0.0;
      // proximity_reward += e.sensed_type === 1 ? 1 - e.sensed_proximity : 0.0;
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
    var reward = proximity_reward + forward_reward + digestion_reward + timed_reward;   
    this.average_reward_window.add(reward);
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

    const geometry = new THREE.BufferGeometry();
    const material = new THREE.LineBasicMaterial( { color: 0xffffff, linewidth: 3 } );    
    const positions = [];
    const colors = [];

    positions.push( 0, 0, 0 );
    positions.push(Math.sin(Math.PI*alpha/180)*r,Math.cos(Math.PI*alpha/180)*r,  Math.PI/2)

    
    geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( positions, 3 ) );
		// geometry.setAttribute( 'color', new THREE.Float32BufferAttribute( colors, 3 ) );
    this._view = new THREE.Line(
      geometry,
      material
    );
    this.end_position = new THREE.Vector3(Math.sin(Math.PI*alpha/180)*r,Math.cos(Math.PI*alpha/180)*r,  Math.PI/2);
    /**setting Eye position and rotation */
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
  get_detection(targets_objs){
    let targets = targets_objs.map((el)=>{
        return el.view;
    });
    let dst = this.a.position.clone();
    // dst.setFromMatrixPosition( this._view.matrixWorld );
    dst.add(this.end_position.clone());
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
  



  /**
   * @class
   * World Contains all features.
   */
export class HuntersWorld {

    /**
     * 
     * @param {*} opt parameters
     * @param opt.algorithm RL algorithm
     * @param opt.width worlds width
     * @param opt.height worlds height
     * @param opt.items_count items count in the world
     * 
     */
    constructor(opt){
      this.params_setter = params_setter.bind(this);

      this._default_params = {
        "items_count": 4000,
        "W": 60,
        "H": 60,
        "D": 60,
        "algorithm": null,
        "UI": null
      };

      this.params_setter(this._default_params, opt);

      this.init();
      this.agents = [];
      this.clock = 0;
      this.walls = []; 

      this.rew_episode = 0;
      this.len_episode = 0;
      this.need_reset_env = 0;
      this.raycaster = new THREE.Raycaster();
      this.bullets = [];

      // set up food and poison
      this.items = []

      for(let k=0;k<this.items_count;k++) {
        this.generateItem();
      }

      if(this.algorithm){
        let agent = new Agent({eyes_count: 10, algo: this.algorithm});
        this.Scene.add(agent.view);
        this.agents.push(agent);        
      }
      this.render();
    }   
    /**
     * 
     * @param {*} agent RL agent algorithm
     */
    addAgent(agent){
      this.Scene.add(agent.view);
      this.agents.push(agent);
      this.n_obs = this.agents[0].get_observation();
    }
    /**
     * generates food and poison
     */
    generateItem(){
      var x = getRandomArbitrary(-this.W, this.W);
      var y = getRandomArbitrary(-this.H, this.H);
      var z = getRandomArbitrary(-this.D, this.D);

      var t = getRandomInt(1, 3); // food or poison (1 and 2)
      if (t == 1){
        var it = new Food(new THREE.Vector3(x, y, z));
      }
      else{
        var it = new Poison(new THREE.Vector3(x, y, z));
      }
      this.items.push(it);
      this.Scene.add(it.view);
    }

  init(json_params){
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



    window.addEventListener("resize", this.onWindowResize.bind(this), false);

    this.Camera.position.set(0,10, 10);
    this.Scene = new THREE.Scene();
    this.Scene.background = new THREE.Color( 0xaaccff );
    this.Scene.fog = new THREE.FogExp2( 0xaaccff, 0.007 );

    let check = false;
    (function(a){if(/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino|android|ipad|playbook|silk/i.test(a)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0,4))) check = true;})(navigator.userAgent||navigator.vendor||window.opera);
    if (check){
      this.CameraObj = new THREE.Object3D();
      this.CameraObj.add(this.Camera);
      this.Controls = new MobileControls({Camera: this.Camera, Object3D: this.CameraObj});      
      this.Scene.add(this.CameraObj);
    }else {
      this.Controls = new FlyControls(this.Camera, document.getElementById("MainContainer"));
      this.Controls.movementSpeed = 13;
      this.Controls.rollSpeed = Math.PI / 8;
      this.Controls.autoForward = false;
      this.Controls.dragToLook = true;  
    }

    this.Loader = new ColladaLoader();

    this.AmbientLight = new THREE.AmbientLight(0xFFFFFF, 0.9);
    this.Scene.add(this.AmbientLight);

    
    this.Clock = new THREE.Clock();

    // if(typeof(document) !== typeof(undefined)){
    //   let TextureLoader = new THREE.TextureLoader();
    //   TextureLoader.load("grass.png", function (tex) {
    //       tex.wrapS = THREE.RepeatWrapping;
    //       tex.wrapT = THREE.RepeatWrapping;
    //       tex.repeat.set(100, 100);
    //       let ground = new THREE.Mesh(new THREE.PlaneBufferGeometry(1000, 1000), new THREE.MeshBasicMaterial({map: tex, side:THREE.DoubleSide}));
    //       ground.rotation.x -= Math.PI/2;
    //       this.Scene.add(ground);
    //   }.bind(this));

    // }
  }
    onWindowResize() {
      this.Camera.aspect = window.innerWidth / window.innerHeight;
      this.Camera.updateProjectionMatrix();
      this.Renderer.setSize(window.innerWidth, window.innerHeight);
    }

    /*Функция проверяет пересечение пуль с уничтожаемыми объектами*/
    controlBulletCollision (bullet, targets_objs)
    {
      let targets = targets_objs.map((el)=>{
          return el.view;
      });
      let dst = new THREE.Vector3();
      dst.setFromMatrixPosition( bullet.view.matrixWorld );
      dst.add(bullet.position.clone().negate());
      dst.normalize();
      this.raycaster.set(bullet._view.position, dst);
      let intersects = this.raycaster.intersectObjects(targets);
      if (intersects.length > 0 && intersects[0].distance < 20){
        return intersects[0].object._rl.obj;
      } else {
        return null;
      }
    }

    reset(){
        this.n_obs = this.agents[0].get_observation();
        this.rew_episode = 0;
        this.len_episode = 0;
        this.clock = 0;
        return this.n_obs.slice()
    }

    get_episode_reward(){
      return this.rew_episode;
    }

    get_episode_length(){
      return this.len_episode;
    }


    render () {
      requestAnimationFrame(this.render.bind(this));
      this.stats.update();
      
      this.Renderer.render(this.Scene, this.Camera);
      var delta = this.Clock.getDelta();
      
      this.Controls.update(delta);
      for (let el of this.bullets){
        el.update(delta);
        let it = this.controlBulletCollision(el, this.items);
        if(it){
          if(it.type === 1) this.agents[0].digestion_signal += it.reward;
          if(it.type === 2) this.agents[0].digestion_signal += it.reward;
          this.removeItem(it);
          this.removeBullet(el);
        }else if (el.way.length() > 20){
          this.removeBullet(el);
          this.agents[0].digestion_signal += -0.99;
        }
      }
    }

    // helper function to get closest colliding walls/items
    computeCollisions(eye, check_walls, check_items) {
      let minres = false;

      if(check_walls) {
          let res = eye.get_detection(this.walls);
          if(res) {
            res.type = 0; // 0 is wall
            if(!minres) { minres=res; }
          }
      }
      // collide with items
      if(check_items) {
        let res = eye.get_detection(this.items);
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

    step(action) {
      let ret = this.tick(action);
      this.rew_episode += ret[1];
      this.len_episode += 1;
      return ret;
    }


    start() {
      requestAnimationFrame(this.start.bind(this));
      this.step();
    }

    addBullet(bullet){
      this.bullets.push(bullet);
      this.Scene.add(bullet.view);
    }
    removeBullet(bullet){
      let index = this.bullets.indexOf(bullet);
      this.bullets.splice(index, 1);
      this.Scene.remove(bullet.view);
    }

    tick(action) {

      this.agents[0].rotX = action[0];
      this.agents[0].rotY = action[1];

      this.agents[0].speed = action[2];  

      if (action[2] > 0.5){
        let bullet = this.agents[0].fire();
        this.addBullet(bullet);
      }
      if (this.need_reset_env){
        this.reset();
        this.need_reset_env = 0;
      }
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
          var res = this.computeCollisions(e, true, true);
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
        states.push(this.agents[i].get_observation());
      }
      
      // apply outputs of agents on evironment
      for(var i=0,n=this.agents.length;i<n;i++) {
        var a = this.agents[i];
        // var v = a.position.clone();
        var v = new THREE.Vector3();
        a._view.getWorldDirection(v);
        v.normalize();
        v.multiplyScalar(a.speed);
        a.position.add(v);
        a.rotation.y += a.rotY;
        a.rotation.x += a.rotX;
        
        var res = this.computeCollisions(a.frontEye, true, false);
        if(res) {
          a.position = a.op;
        }
        
        // handle boundary conditions
        if(a.position.x< -this.W/2)a.position.x=-this.W/2;
        if(a.position.x>this.W/2)a.position.x=this.W/2;
        if(a.position.z< -this.H/2)a.position.z=-this.H/2;
        if(a.position.z>this.H/2)a.position.z=this.H/2;
        if(a.position.y< -this.D/2)a.position.y=-this.D/2;
        if(a.position.y>this.D/2)a.position.y=this.D/2;

      }
      
      for(var i=0,n=this.items.length;i<n;i++) {
        var it = this.items[i];
        it.age += 1;
        
        // see if some agent gets lunch
        for(var j=0,m=this.agents.length;j<m;j++) {
          var a = this.agents[j];
          var d = a.position.distanceTo(it.position);
          if(d < it.rad + a.rad) {
            
            var rescheck = this.computeCollisions(a.frontEye, true, false);
            if(!rescheck) { 
              if(it.type === 1) a.digestion_signal += it.reward;
              if(it.type === 2) a.digestion_signal += it.reward;
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
      if(this.items.length < this.items_count) {
        this.generateItem();
      }
      let rewards = [];
      for(var i=0,n=this.agents.length;i<n;i++) {
        rewards.push(this.agents[i].get_reward());
      }
      done = 0;

      state = states[0];
      reward = rewards[0];
      let info = {};
      
      let ret_data = [state, reward, done, info];
      if(this.clock % 1000 == 0){
        done = true;
        ret_data[2] = done;
        this.need_reset_env = 1;
      }
      return ret_data; 
    }
  }
  
